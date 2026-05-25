"""Web dashboard for polymarket-bot.

A single Flask app that:
  - Serves a Chinese-language dashboard at GET /
  - Manages the trading bot subprocess (start/stop/status) via JSON APIs
  - Reads logs/trades.csv and logs/ticks.csv for live position + equity data

Run it with:
    ./run-web.sh
or:
    python -m web.app

Then open http://localhost:8000 in your browser.

The bot itself runs as a subprocess of this server. Stopping the server
also stops the bot. State (positions, equity) lives in the bot subprocess
and is observed via the CSV journal.
"""
from __future__ import annotations

import csv
import logging
import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from flask import Flask, jsonify, render_template

# polymarket-bot project root (one above this web/ folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

log = logging.getLogger("polymarket_bot.web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ---------- Bot subprocess manager ----------

class BotManager:
    """Owns the bot subprocess. Captures its stdout into an in-memory ring
    buffer so the dashboard can show recent log lines without tailing files.
    """

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.process: Optional[subprocess.Popen] = None
        self.log_lines: Deque[str] = deque(maxlen=400)
        self._reader: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ----- queries -----
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def get_status(self) -> Dict[str, Any]:
        if self.process is None:
            return {"running": False, "pid": None, "exit_code": None}
        rc = self.process.poll()
        return {"running": rc is None, "pid": self.process.pid, "exit_code": rc}

    def get_log(self, n: int = 200) -> List[str]:
        return list(self.log_lines)[-n:]

    # ----- mutations -----
    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "Bot 已经在运行中"
            # Prefer the project venv python if present (matches what run.sh uses).
            venv_py = self.root / ".venv" / "bin" / "python"
            py_bin = str(venv_py) if venv_py.exists() else "python3"
            try:
                self.process = subprocess.Popen(
                    [py_bin, "-u", "-m", "src.main"],
                    cwd=str(self.root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                    # New process group so SIGTERM hits the whole tree.
                    preexec_fn=os.setsid,
                )
            except FileNotFoundError as e:
                return False, f"找不到 Python: {e}. 请先运行 ./run.sh --check 安装依赖。"
            except Exception as e:
                return False, f"启动失败: {e}"
            self._reader = threading.Thread(target=self._drain_output, daemon=True)
            self._reader.start()
            self.log_lines.append(f"[web] Bot 子进程已启动 (pid={self.process.pid})")
            log.info("Started bot subprocess pid=%d", self.process.pid)
            return True, f"已启动 (PID={self.process.pid})"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self.is_running():
                return False, "Bot 未运行"
            assert self.process is not None
            pid = self.process.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log.warning("Bot did not exit on SIGTERM, sending SIGKILL")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self.log_lines.append(f"[web] Bot 子进程已停止 (pid={pid})")
            log.info("Stopped bot subprocess pid=%d", pid)
            self.process = None
            return True, "已停止"

    def _drain_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        try:
            for line in self.process.stdout:
                self.log_lines.append(line.rstrip("\n"))
        except (ValueError, OSError):
            # stdout closed during shutdown — normal.
            pass


# ---------- CSV readers ----------

def tail_csv(path: Path, n: int) -> List[Dict[str, str]]:
    """Read last `n` rows of a CSV with headers. Returns [] on missing file."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    return rows[-n:]


def equity_series(n: int = 300) -> List[Dict[str, Any]]:
    rows = tail_csv(LOGS_DIR / "ticks.csv", n)
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            eq = float(r.get("equity_usd", "0") or 0)
        except ValueError:
            continue
        if eq <= 0:
            continue
        out.append({"t": r.get("timestamp_utc", ""), "equity": eq})
    return out


def latest_tick() -> Optional[Dict[str, Any]]:
    rows = tail_csv(LOGS_DIR / "ticks.csv", 1)
    return rows[-1] if rows else None


def recent_trades(n: int = 30) -> List[Dict[str, Any]]:
    return list(reversed(tail_csv(LOGS_DIR / "trades.csv", n)))


def recent_ticks(n: int = 30) -> List[Dict[str, Any]]:
    return list(reversed(tail_csv(LOGS_DIR / "ticks.csv", n)))


def decision_counts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for r in rows:
        # Simplify "HOLD_scout_too_late(204s)" -> "HOLD_scout_too_late"
        key = (r.get("decision") or "").split("(")[0].rstrip("_")
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]


# ---------- Flask app ----------

app = Flask(__name__, template_folder="templates")
bot = BotManager(PROJECT_ROOT)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    tick = latest_tick()
    payload: Dict[str, Any] = {
        "bot": bot.get_status(),
        "market": None,
        "spot": None,
        "position": None,
        "decision": tick.get("decision") if tick else "",
        "tick_time": tick.get("timestamp_utc") if tick else None,
    }
    if tick:
        try:
            yes = float(tick.get("yes_price") or 0)
            no = float(tick.get("no_price") or 0)
        except ValueError:
            yes = no = 0.0
        payload["market"] = {
            "slug": tick.get("market_slug", ""),
            "yes_price": yes,
            "no_price": no,
            "total": round(yes + no, 4),
            "remaining_sec": float(tick.get("remaining_sec") or 0),
        }
        payload["spot"] = {
            "last": float(tick.get("spot_last") or 0),
            "short_return": float(tick.get("spot_short_return") or 0),
            "long_return": float(tick.get("spot_long_return") or 0),
            "rsi": float(tick.get("spot_rsi") or 0),
            "volume_ratio": float(tick.get("spot_volume_ratio") or 0),
            "momentum_score": float(tick.get("momentum_score") or 0),
        }
        payload["position"] = {
            "cash": float(tick.get("cash_usd") or 0),
            "yes_shares": float(tick.get("yes_shares") or 0),
            "no_shares": float(tick.get("no_shares") or 0),
            "equity": float(tick.get("equity_usd") or 0),
        }
    return jsonify(payload)


@app.route("/api/equity")
def api_equity():
    return jsonify({"points": equity_series(n=300)})


@app.route("/api/trades")
def api_trades():
    return jsonify({"trades": recent_trades(n=30)})


@app.route("/api/ticks")
def api_ticks():
    return jsonify({"ticks": recent_ticks(n=30)})


@app.route("/api/decisions")
def api_decisions():
    rows = tail_csv(LOGS_DIR / "ticks.csv", 500)
    return jsonify({"counts": decision_counts(rows), "total": len(rows)})


@app.route("/api/log")
def api_log():
    return jsonify({"lines": bot.get_log(n=200)})


@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    ok, msg = bot.start()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    ok, msg = bot.stop()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "logs_dir": str(LOGS_DIR), "exists": LOGS_DIR.exists()})


def main() -> None:
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    log.info("Polymarket-bot dashboard at http://%s:%d", host, port)
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Logs dir:     %s", LOGS_DIR)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
