"""
DERIV NO TOUCH BOT — R_10
==========================
Symbol   : R_10 (Volatility 10 Index)
Contract : NOTOUCH — win if price never touches the barrier
           during the contract period.

Strategy
--------
  Enter only during calm σ windows (σ < session average).
  Set barrier at BARRIER_MULT × rolling σ away from current price.
  Duration is adaptive:
    · σ very calm  (< 0.7 × avg) → 10 ticks (barrier has more time to hold)
    · σ calm       (< 1.0 × avg) → 7 ticks
    · σ elevated   (≥ 1.0 × avg) → skip, too risky

  Monitor data basis (R_10, ~3100 ticks):
    · σ avg = 0.089, range 0.06–0.12
    · Symbol spends 99.9% of time in NORMAL band
    · Barrier at 3× σ ≈ 0.267 — well outside typical tick movement

Risk
----
  Martingale 1.5× on losses, reset after MAX_LOSSES or any win.
  Circuit breaker: 3 consecutive losses → 5 min pause.
  Session target: +$10 | Session stop: -$20
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")


# ============================================================================
# CONFIGURATION
# ============================================================================

def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


CONFIG = {
    # Deriv credentials
    "api_token":          _env("DERIV_API_TOKEN", "REPLACE_WITH_YOUR_TOKEN"),
    "app_id":             _env("DERIV_APP_ID", 1089),
    "symbol":             _env("SYMBOL", "R_10"),
    "currency":           "USD",

    # Signal — rolling window for σ
    "vol_window":         _env("VOL_WINDOW", 50),
    "min_warmup":         _env("MIN_WARMUP", 60),     # ticks before trading

    # Entry gate — only trade when σ < ENTRY_SIGMA_RATIO × session avg
    "entry_sigma_ratio":  _env("ENTRY_SIGMA_RATIO", 1.0),

    # Barrier placement
    "barrier_mult":       _env("BARRIER_MULT", 3.0),  # × current σ

    # Adaptive duration thresholds
    "dur_calm_threshold": _env("DUR_CALM_THRESH", 0.7),  # σ < 0.7×avg → 10t
    "dur_calm_ticks":     _env("DUR_CALM_TICKS", 10),
    "dur_normal_ticks":   _env("DUR_NORMAL_TICKS", 7),

    # Cooldown between trades (ticks)
    "cooldown_ticks":     _env("COOLDOWN_TICKS", 5),

    # Directional momentum for barrier placement
    "momentum_window":    _env("MOMENTUM_WINDOW", 10),   # last N ticks
    "momentum_threshold": _env("MOMENTUM_THRESH", 0.05), # net drift to count as directional

    # Risk / Martingale
    "initial_stake":      _env("INITIAL_STAKE", 0.35),
    "martingale_mul":     _env("MARTINGALE_MUL", 1.50),
    "max_losses":         _env("MAX_LOSSES", 4),
    "target_profit":      _env("TARGET_PROFIT", 10.0),
    "stop_loss":          _env("STOP_LOSS", 20.0),

    # Circuit breaker
    "cb_limit":           _env("CB_LIMIT", 3),
    "cb_pause_secs":      _env("CB_PAUSE", 300),

    # Resilience
    "lock_timeout":       _env("LOCK_TIMEOUT", 60),
    "buy_retries":        _env("BUY_RETRIES", 8),
    "reconnect_min":      _env("RECONNECT_MIN", 2),
    "reconnect_max":      _env("RECONNECT_MAX", 60),
    "ws_ping":            _env("WS_PING", 30),
    "orphan_attempts":    _env("ORPHAN_ATTEMPTS", 4),
    "orphan_interval":    _env("ORPHAN_INTERVAL", 3),
}


# ============================================================================
# HELPERS
# ============================================================================

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(tag, msg):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)

def _jlog(obj):
    print(json.dumps(obj), flush=True)


# ============================================================================
# VOLATILITY ENGINE
# ============================================================================

class VolEngine:
    def __init__(self, cfg):
        self.cfg           = cfg
        self.prices        = deque(maxlen=cfg["vol_window"] + 2)
        self.moves         = deque(maxlen=cfg["vol_window"])
        self.tick_n        = 0
        self.sigma_history = deque(maxlen=500)
        # Directional momentum: store raw signed moves
        self.signed_moves  = deque(maxlen=cfg["momentum_window"])

    def add_tick(self, price: float):
        if self.prices:
            raw  = price - self.prices[-1]
            self.moves.append(abs(raw))
            self.signed_moves.append(raw)
        self.prices.append(price)
        self.tick_n += 1

    def is_ready(self) -> bool:
        return self.tick_n >= self.cfg["min_warmup"]

    def sigma(self) -> float:
        if len(self.moves) < 5:
            return 0.0
        moves = list(self.moves)
        mu    = sum(moves) / len(moves)
        var   = sum((x - mu) ** 2 for x in moves) / len(moves)
        s     = math.sqrt(var)
        self.sigma_history.append(s)
        return s

    def session_avg_sigma(self) -> float:
        if not self.sigma_history:
            return 0.089
        return sum(self.sigma_history) / len(self.sigma_history)

    def current_price(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None

    def momentum_bias(self) -> str:
        """
        Returns 'UP', 'DOWN', or 'NEUTRAL' based on net direction
        of the last momentum_window ticks.
        UP   → price drifting up   → place barrier ABOVE (bet it won't go higher)
        DOWN → price drifting down → place barrier BELOW (bet it won't go lower)
        NEUTRAL → use geometric (further side)
        """
        if len(self.signed_moves) < self.cfg["momentum_window"]:
            return "NEUTRAL"
        net = sum(self.signed_moves)
        threshold = self.cfg["momentum_threshold"]
        if net > threshold:
            return "UP"
        if net < -threshold:
            return "DOWN"
        return "NEUTRAL"

    def evaluate(self):
        """
        Returns (should_trade, barrier_str, duration, sigma, bias)
        barrier_str: e.g. '+0.27' or '-0.31' — relative to current price.
        """
        if not self.is_ready():
            return False, "", 0, 0, "NEUTRAL"

        s     = self.sigma()
        avg   = self.session_avg_sigma()
        price = self.current_price()

        if price is None or avg == 0:
            return False, "", 0, s, "NEUTRAL"

        ratio = s / avg if avg > 0 else 1.0

        # Entry gate
        if ratio >= self.cfg["entry_sigma_ratio"]:
            return False, "", 0, s, "NEUTRAL"

        # Adaptive duration
        if ratio < self.cfg["dur_calm_threshold"]:
            duration = self.cfg["dur_calm_ticks"]
        else:
            duration = self.cfg["dur_normal_ticks"]

        # Barrier distance (2 dp max for Deriv)
        barrier_dist = round(self.cfg["barrier_mult"] * s, 2)
        barrier_dist = max(barrier_dist, 0.05)

        # Directional bias — place barrier in the direction of momentum
        # (bet the momentum won't continue far enough to touch it)
        bias = self.momentum_bias()

        if bias == "UP":
            # Trending up → place barrier above, bet it won't spike further
            barrier_str = f"+{barrier_dist:.2f}"
        elif bias == "DOWN":
            # Trending down → place barrier below, bet it won't drop further
            barrier_str = f"-{barrier_dist:.2f}"
        else:
            # Neutral — use the safer side (larger distance from price)
            barrier_str = f"+{barrier_dist:.2f}"

        return True, barrier_str, duration, s, bias


# ============================================================================
# MARTINGALE MANAGER
# ============================================================================

class MartingaleManager:
    def __init__(self, cfg):
        self.cfg           = cfg
        self.initial_stake = cfg["initial_stake"]
        self.current_stake = cfg["initial_stake"]
        self.mul           = cfg["martingale_mul"]
        self.max_losses    = cfg["max_losses"]
        self.target_profit = cfg["target_profit"]
        self.stop_loss     = cfg["stop_loss"]
        self.loss_streak   = 0
        self.total_profit  = 0.0
        self.wins          = 0
        self.losses        = 0

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def record_win(self, profit: float):
        self.wins         += 1
        self.total_profit += profit
        self.loss_streak   = 0
        self.current_stake = self.initial_stake
        _log("WIN", f"+${profit:.2f} | stake reset → ${self.initial_stake:.2f}")
        self._stats()

    def record_loss(self, loss: float):
        self.losses       += 1
        self.total_profit += loss
        self.loss_streak  += 1
        _log("LOSS", f"-${abs(loss):.2f} | streak={self.loss_streak}")
        if self.loss_streak >= self.max_losses:
            _log("MARTI", f"{self.max_losses} losses → reset to ${self.initial_stake:.2f}")
            self.current_stake = self.initial_stake
            self.loss_streak   = 0
        else:
            self.current_stake = round(self.current_stake * self.mul, 2)
            _log("MARTI", f"L{self.loss_streak} next stake ${self.current_stake:.2f}")
        self._stats()

    def can_trade(self) -> bool:
        if self.total_profit >= self.target_profit:
            _log("RISK", f"Target profit reached (${self.total_profit:.2f}) — stopping")
            return False
        if self.total_profit <= -self.stop_loss:
            _log("RISK", f"Stop-loss hit (${self.total_profit:.2f}) — stopping")
            return False
        return True

    def _stats(self):
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0.0
        print(f"\n{'='*55}", flush=True)
        print(f"  {total} trades | W:{self.wins} L:{self.losses} | WR:{wr:.1f}%", flush=True)
        print(f"  P&L ${self.total_profit:+.2f} | next stake ${self.current_stake:.2f}", flush=True)
        print(f"{'='*55}\n", flush=True)
        _jlog({
            "type": "stats", "trades": total, "wins": self.wins,
            "losses": self.losses, "wr": round(wr, 1),
            "pnl": round(self.total_profit, 2),
            "next_stake": self.current_stake, "ts": _ts(),
        })


# ============================================================================
# DERIV CLIENT
# ============================================================================

class DerivClient:
    def __init__(self, cfg):
        self.cfg      = cfg
        self.endpoint = (
            f"wss://ws.derivws.com/websockets/v3?app_id={cfg['app_id']}"
        )
        self.ws              = None
        self._send_queue     = None
        self._inbox          = None
        self._send_task      = None
        self._recv_task      = None

    async def connect(self) -> bool:
        _log("WS", f"Connecting → {self.endpoint}")
        self.ws = await websockets.connect(
            self.endpoint,
            ping_interval=self.cfg["ws_ping"],
            ping_timeout=20,
            close_timeout=10,
        )
        self._send_queue = asyncio.Queue()
        self._inbox      = asyncio.Queue()
        self._start_io()
        await self._send({"authorize": self.cfg["api_token"]})
        resp = await self._recv_type("authorize", timeout=15)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("AUTH", f"Failed: {err}")
            return False
        auth = resp.get("authorize", {})
        _log("AUTH",
             f"OK | {auth.get('loginid','?')} | "
             f"Balance: ${auth.get('balance', 0):.2f}")
        return True

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            _log("RECV", f"Error: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send(self, data):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def receive(self, timeout=60):
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def _recv_type(self, msg_type, timeout=10):
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            _log("BALANCE", f"Fetch error: {exc}")
        return None

    async def subscribe_ticks(self) -> bool:
        sym = self.cfg["symbol"]
        await self._send({"ticks": sym, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("TICK", f"Subscribe failed: {err}")
            return False
        _log("TICK", f"Subscribed to {sym}")
        return True

    async def place_notouch(
            self, barrier_str: str,
            duration: int, stake: float) -> Optional[str]:
        """
        Place a NOTOUCH contract with a pre-computed relative barrier string.
        barrier_str examples: '+0.27', '-0.31'
        Direction is chosen by the VolEngine based on momentum bias.
        """
        proposal_req = {
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": "NOTOUCH",
            "currency":      self.cfg["currency"],
            "duration":      duration,
            "duration_unit": "t",
            "symbol":        self.cfg["symbol"],
            "barrier":       barrier_str,
        }

        await self._send(proposal_req)
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            err = (proposal or {}).get("error", {}).get("message", "timeout")
            _log("PROPOSAL", f"Error: {err}")
            return None

        prop        = proposal.get("proposal", {})
        pid         = prop.get("id")
        ask         = float(prop.get("ask_price", stake))
        payout      = float(prop.get("payout", 0))
        if not pid:
            _log("PROPOSAL", "No proposal ID")
            return None

        roi = ((payout - ask) / ask * 100) if ask > 0 else 0
        _log("PROPOSAL",
             f"NOTOUCH {duration}t  barrier={barrier_str}  "
             f"ask=${ask:.2f}  payout=${payout:.2f}  ROI={roi:.1f}%")

        buy_time    = time.time()
        contract_id = None
        await self._send({"buy": pid, "price": ask})

        for attempt in range(self.cfg["buy_retries"]):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                _log("BUY", f"No response (attempt {attempt + 1})")
                continue
            if "error" in resp:
                _log("BUY", f"Error: {resp['error'].get('message', '')}")
                return None
            contract_id = resp.get("buy", {}).get("contract_id")
            if contract_id:
                break

        if not contract_id:
            _log("BUY", "No contract_id — orphan recovery")
            contract_id = await self._recover_orphan(stake, buy_time)
            if contract_id:
                _log("BUY", f"Orphan recovered → {contract_id}")
            else:
                _log("BUY", "Orphan recovery failed")
                return None

        _log("TRADE",
             f"NOTOUCH  ${stake:.2f}  {duration}t  "
             f"barrier={barrier_str}  contract={contract_id}")

        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1,
            })
        except Exception:
            pass

        return str(contract_id)

    async def _recover_orphan(self, stake, buy_time) -> Optional[str]:
        for attempt in range(self.cfg["orphan_attempts"]):
            await asyncio.sleep(self.cfg["orphan_interval"])
            try:
                await self._send({"profit_table": 1, "description": 1,
                                  "sort": "DESC", "limit": 5})
                resp = await self._recv_type("profit_table", timeout=10)
                if not resp or "error" in resp:
                    continue
                for tx in resp.get("profit_table", {}).get("transactions", []):
                    if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                            float(tx.get("purchase_time", 0)) >= buy_time - 5):
                        return str(tx.get("contract_id"))
            except Exception as exc:
                _log("ORPHAN", f"Poll {attempt + 1} error: {exc}")
        return None

    async def poll_contract(self, contract_id) -> Optional[dict]:
        try:
            await self._send({"proposal_open_contract": 1,
                              "contract_id": contract_id})
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            _log("POLL", f"Error: {exc}")
        return None


# ============================================================================
# MAIN BOT
# ============================================================================

class NoTouchBot:
    def __init__(self):
        self.cfg    = CONFIG
        self.client = DerivClient(CONFIG)
        self.engine = VolEngine(CONFIG)
        self.risk   = MartingaleManager(CONFIG)

        self.tick_n:             int   = 0
        self._last_trade_tick:   int   = 0
        self.current_contract:   Optional[dict] = None
        self.waiting_for_result: bool  = False
        self.lock_since:         Optional[float] = None
        self._evaluating:        bool  = False
        self._balance_before:    Optional[float] = None
        self._cb_paused_until:   float = 0.0
        self._stop:              bool  = False

    def _unlock(self, reason="manual"):
        if self.waiting_for_result:
            cid = (self.current_contract or {}).get("id", "?")
            _log("UNLOCK", f"Contract {cid} ({reason})")
        self.waiting_for_result = False
        self.current_contract   = None
        self.lock_since         = None
        self._evaluating        = False

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        exp     = (self.current_contract or {}).get("duration", 10)
        timeout = exp + self.cfg["lock_timeout"]
        if time.monotonic() - self.lock_since >= timeout:
            _log("TIMEOUT", f"Auto-unlocking after {timeout}s")
            self._unlock("timeout")

    @staticmethod
    def _is_settled(data) -> bool:
        if data.get("is_settled"):
            return True
        for key in ("status", "contract_status"):
            if data.get(key, "").lower() in ("sold", "won", "lost"):
                return True
        return False

    async def handle_settlement(self, data) -> Optional[bool]:
        cid = str(data.get("contract_id", ""))
        if not self.current_contract or cid != self.current_contract["id"]:
            return None
        if not self._is_settled(data):
            return None

        bal_after  = await self.client.fetch_balance()
        api_profit = float(data.get("profit", 0))
        status     = data.get("status", "unknown")

        if bal_after is not None and self._balance_before is not None:
            actual = round(bal_after - self._balance_before, 2)
            _log("BALANCE",
                 f"Pre: ${self._balance_before:.2f} → "
                 f"Post: ${bal_after:.2f} | "
                 f"Actual: ${actual:+.2f} | API: ${api_profit:+.2f}")
        else:
            actual = api_profit

        print(f"\nRESULT  contract={cid}  status={status}  "
              f"profit=${actual:+.2f}", flush=True)

        if actual > 0:
            self.risk.record_win(actual)
        else:
            self.risk.record_loss(actual)
            if (self.risk.loss_streak > 0 and
                    self.risk.loss_streak % self.cfg["cb_limit"] == 0):
                pause = self.cfg["cb_pause_secs"]
                self._cb_paused_until = time.monotonic() + pause
                _log("BREAKER",
                     f"{self.cfg['cb_limit']} consecutive losses → "
                     f"pausing {pause}s ({pause // 60}m)")

        _jlog({
            "type": "result", "cid": cid, "status": status,
            "profit": actual, "pnl": self.risk.total_profit,
            "wins": self.risk.wins, "losses": self.risk.losses,
            "ts": _ts(),
        })

        self._balance_before = None
        self._unlock("settlement")
        return self.risk.can_trade()

    async def on_tick(self, price: float):
        self.tick_n += 1
        self.cfg["_last_price"] = price
        self.engine.add_tick(price)
        self._check_lock_timeout()

        if self.tick_n % 20 == 0:
            warmup_left = max(0, self.cfg["min_warmup"] - self.engine.tick_n)
            status = ("WAIT" if self.waiting_for_result else
                      f"WARMUP({warmup_left})" if warmup_left > 0 else "READY")
            print(f"\r  #{self.tick_n}  p={price:.5f}  {status}  {_ts()}",
                  end="", flush=True)

        if self.waiting_for_result or self._evaluating:
            return
        if not self.engine.is_ready():
            return
        if (self.tick_n - self._last_trade_tick) < self.cfg["cooldown_ticks"]:
            return

        self._evaluating = True
        try:
            await self._evaluate(price)
        finally:
            self._evaluating = False

    async def _evaluate(self, price: float):
        if self.waiting_for_result:
            return

        ok, barrier_str, duration, sigma, bias = self.engine.evaluate()
        avg = self.engine.session_avg_sigma()

        print(f"\n{'='*55}", flush=True)
        print(f"SIGNAL  #{self.tick_n}  {_ts()}", flush=True)
        ratio_str = f"{sigma/avg:.2f}" if avg > 0 else "?"
        print(f"  σ={sigma:.6f}  avg={avg:.6f}  ratio={ratio_str}  bias={bias}",
              flush=True)

        if not ok:
            print(f"  → No trade (σ elevated)", flush=True)
            print(f"{'='*55}", flush=True)
            return

        print(f"  → NOTOUCH  barrier={barrier_str}  "
              f"duration={duration}t  bias={bias}", flush=True)
        print(f"{'='*55}", flush=True)

        now = time.monotonic()
        if now < self._cb_paused_until:
            remaining = self._cb_paused_until - now
            _log("BREAKER", f"Paused — {remaining:.0f}s remaining")
            return

        if not self.risk.can_trade():
            return

        stake = self.risk.get_stake()

        bal = await self.client.fetch_balance()
        if bal is not None:
            self._balance_before = bal
            _log("BALANCE", f"Pre-trade: ${bal:.2f}")
        else:
            self._balance_before = None

        contract_id = await self.client.place_notouch(
            barrier_str, duration, stake)

        if contract_id:
            self.current_contract = {
                "id":       contract_id,
                "stake":    stake,
                "duration": duration,
                "sigma":    sigma,
                "bias":     bias,
                "barrier":  barrier_str,
                "time":     datetime.now(),
            }
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            self._last_trade_tick   = self.tick_n
            _log("LOCK", f"Waiting for result on {contract_id}")
            _jlog({
                "type":    "trade",
                "cid":     contract_id,
                "duration": duration,
                "stake":   stake,
                "sigma":   round(sigma, 6),
                "barrier": barrier_str,
                "bias":    bias,
                "ts":      _ts(),
            })
        else:
            self._balance_before = None
            _log("TRADE", "Placement failed — ready for next signal")

    async def _reconnect(self) -> bool:
        delay   = self.cfg["reconnect_min"]
        attempt = 0
        while not self._stop:
            attempt += 1
            _log("RECONNECT", f"Attempt {attempt} in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg["reconnect_max"])
            await self.client.close()
            self.client = DerivClient(self.cfg)
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                if self.waiting_for_result and self.current_contract:
                    cid  = self.current_contract["id"]
                    data = await self.client.poll_contract(cid)
                    if data:
                        await self.handle_settlement(data)
                    if self.waiting_for_result:
                        await self.client._send({
                            "proposal_open_contract": 1,
                            "contract_id": cid, "subscribe": 1,
                        })
                _log("RECONNECT", "OK")
                return True
            except Exception as exc:
                _log("RECONNECT", f"Error: {exc}")
        return False

    async def _console(self):
        loop = asyncio.get_event_loop()
        _log("CMD", "Commands: [s]tats  [u]nlock  [q]uit")
        while not self._stop:
            try:
                cmd = (await loop.run_in_executor(None, input)).strip().lower()
                if cmd == "s":
                    self.risk._stats()
                elif cmd == "u":
                    self._unlock("user command")
                elif cmd in ("q", "quit", "exit"):
                    self._stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break

    async def run(self):
        cfg = self.cfg
        print("\n" + "="*55, flush=True)
        print("  DERIV NO TOUCH BOT", flush=True)
        print("="*55, flush=True)
        print(f"  Symbol   : {cfg['symbol']}", flush=True)
        print(f"  Barrier  : {cfg['barrier_mult']}× rolling σ", flush=True)
        print(f"  Duration : adaptive ({cfg['dur_normal_ticks']}t normal, "
              f"{cfg['dur_calm_ticks']}t calm)", flush=True)
        print(f"  Stake    : ${cfg['initial_stake']:.2f} "
              f"(×{cfg['martingale_mul']} mart, "
              f"reset @{cfg['max_losses']} losses)", flush=True)
        print(f"  Target   : +${cfg['target_profit']}  "
              f"Stop: -${cfg['stop_loss']}", flush=True)
        print(f"  Breaker  : {cfg['cb_limit']} losses → "
              f"{cfg['cb_pause_secs']}s pause", flush=True)
        print(f"  Warmup   : {cfg['min_warmup']} ticks", flush=True)
        print("="*55 + "\n", flush=True)

        if cfg["api_token"] in ("REPLACE_WITH_YOUR_TOKEN", ""):
            _log("ERROR", "Set DERIV_API_TOKEN before running")
            return

        if not await self.client.connect():
            return
        if not await self.client.subscribe_ticks():
            return

        _log("BOT", f"Live — warming up ({cfg['min_warmup']} ticks)...")
        console_task = asyncio.create_task(self._console())

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    _log("WS", "Disconnected — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        _log("WS", "Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    quote = response["tick"].get("quote")
                    if quote is not None:
                        print()
                        await self.on_tick(float(quote))

                if "proposal_open_contract" in response:
                    result = await self.handle_settlement(
                        response["proposal_open_contract"])
                    if result is False:
                        break

                if "buy" in response:
                    result = await self.handle_settlement(response["buy"])
                    if result is False:
                        break

                if "transaction" in response:
                    tx = response["transaction"]
                    if "contract_id" in tx:
                        result = await self.handle_settlement({
                            "contract_id": tx.get("contract_id"),
                            "profit":      tx.get("profit", 0),
                            "status":      tx.get("action", ""),
                            "is_settled":  True,
                        })
                        if result is False:
                            break

        except KeyboardInterrupt:
            print("\n\nInterrupted", flush=True)
        except Exception as exc:
            print(f"\nUnhandled error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            console_task.cancel()
            await self.client.close()
            print("\nFINAL STATS", flush=True)
            self.risk._stats()
            print("Goodbye", flush=True)


async def main():
    bot = NoTouchBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
