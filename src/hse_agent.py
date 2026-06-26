"""
HSE Multi-Agent System — Autonomous Incident Response

Architecture:
  HSEOrchestrator (coordinator)
      ├── FallResponseAgent    — handles SLIP_FALL events
      ├── PanicResponseAgent   — handles RUNNING_PANIC events
      └── EscalationAgent      — monitors repeat incidents, triggers escalation

  Shared: IncidentMemory — real-time + historical pattern analysis

No external APIs. No LLMs. Pure agentic reasoning.
"""

import smtplib
import time
import datetime
import logging
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.incident_memory import IncidentMemory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  BASE AGENT — shared email tool + SMTP
# ─────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, email_cfg: dict):
        self._cfg = email_cfg

    def _send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._cfg["sender_email"]
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))
        try:
            with smtplib.SMTP(self._cfg["smtp_host"], self._cfg["smtp_port"]) as s:
                s.ehlo(); s.starttls()
                s.login(self._cfg["sender_email"], self._cfg["sender_app_password"])
                s.sendmail(self._cfg["sender_email"], to_email, msg.as_string())
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error(
                "[SMTP] Auth failed. Enable 2-Step Verification and use an App Password.\n"
                "  → https://myaccount.google.com/apppasswords"
            )
        except smtplib.SMTPException as e:
            logger.error("[SMTP] Send failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────
#  FALL RESPONSE AGENT
#  Speciality: medical emergency response for SLIP_FALL
# ─────────────────────────────────────────────────────────

class FallResponseAgent(BaseAgent):
    """
    Autonomous agent for slip/fall medical emergency response.
    Decides: who to notify, what action to recommend, based on context + memory.
    """

    def respond(self, track_id: int, memory: IncidentMemory, context: dict):
        guard = self._cfg["guard"]
        supervisor = self._cfg["supervisor"]

        # Agent queries memory to personalize response
        count_today = memory.get_incident_count(track_id, "SLIP_FALL", within_seconds=86400)
        is_repeat = memory.is_repeat_offender(track_id, "SLIP_FALL", threshold=2, within_seconds=3600)
        today_stats = memory.get_today_summary()

        repeat_note = (
            f"⚠️ REPEAT INCIDENT: This person (ID #{track_id}) has fallen "
            f"<strong>{count_today} times today</strong>. Immediate medical review is recommended."
            if is_repeat else
            f"First recorded fall for ID #{track_id} today."
        )

        # Guard email — short, action-focused
        guard_html = _build_guard_email(
            name=guard["name"], color="#cc0000", emoji="🚨",
            severity="CRITICAL", category="Medical Emergency / Slip & Fall",
            track_id=track_id, context=context,
            action="Respond immediately. Check consciousness. Call medical support if unresponsive.",
            repeat_note=repeat_note,
        )
        ok = self._send_email(guard["email"],
            f"🚨 CRITICAL: Fall Detected | Person ID #{track_id} | {context['timestamp']}",
            guard_html)
        if ok:
            logger.info("[FallResponseAgent] 📩 Guard notified → %s", guard["email"])

        # Supervisor email — full incident intelligence report
        supervisor_html = _build_supervisor_email(
            name=supervisor["name"], color="#cc0000", emoji="🚨",
            severity="CRITICAL", category="Slip & Fall",
            alert_type="SLIP_FALL", track_id=track_id, context=context,
            reasoning=(
                f"{repeat_note} "
                f"Today's session has recorded <strong>{today_stats['total']} total incidents</strong> "
                f"with peak activity at <strong>{today_stats['peak_hour']}</strong>."
            ),
            action="Verify medical response. Log incident in HSE register within 2 hours. "
                   "Review area for slip hazards (wet floor, cables, uneven surface).",
            today_stats=today_stats,
        )
        ok = self._send_email(supervisor["email"],
            f"[HSE REPORT] CRITICAL Fall Incident — ID #{track_id} | {context['timestamp']}",
            supervisor_html)
        if ok:
            logger.info("[FallResponseAgent] 📧 Supervisor report sent → %s", supervisor["email"])

        # IoT buzzer
        _trigger_buzzer("SLIP_FALL")


# ─────────────────────────────────────────────────────────
#  PANIC RESPONSE AGENT
#  Speciality: running/evacuation risk response
# ─────────────────────────────────────────────────────────

class PanicResponseAgent(BaseAgent):
    """
    Autonomous agent for running/panic behavior response.
    Office hours → notify guard only.
    Night shift → escalate to supervisor immediately.
    Repeat pattern → flag as potential evacuation risk.
    """

    def respond(self, track_id: int, memory: IncidentMemory, context: dict):
        guard = self._cfg["guard"]
        supervisor = self._cfg["supervisor"]

        count_session = memory.get_incident_count(track_id, "RUNNING_PANIC", within_seconds=600)
        today_stats = memory.get_today_summary()

        # Agent reasons: multiple running alerts in 10 min = possible evacuation
        if count_session >= 3:
            risk_label = "🔴 EVACUATION RISK — Multiple running alerts in 10 minutes"
            action = ("Initiate evacuation check. Multiple persons may be running. "
                      "Check for fire, gas leak, or physical altercation.")
        else:
            risk_label = "Isolated running behavior detected"
            action = "Proceed to flagged zone. Check for hazard or panic trigger."

        guard_html = _build_guard_email(
            name=guard["name"], color="#e65c00", emoji="⚠️",
            severity="HIGH", category="Panic / Running Behavior",
            track_id=track_id, context=context,
            action=action,
            repeat_note=risk_label,
        )
        ok = self._send_email(guard["email"],
            f"⚠️ HIGH ALERT: Running/Panic Detected | Person ID #{track_id} | {context['timestamp']}",
            guard_html)
        if ok:
            logger.info("[PanicResponseAgent] 📩 Guard notified → %s", guard["email"])

        # Supervisor only on night shift or repeated panic
        if context["shift"] == "NIGHT_SHIFT" or count_session >= 3:
            reasoning = (
                f"{'Night shift escalation: reduced staffing on-site. ' if context['shift'] == 'NIGHT_SHIFT' else ''}"
                f"{risk_label}. "
                f"Today total incidents: <strong>{today_stats['total']}</strong>."
            )
            supervisor_html = _build_supervisor_email(
                name=supervisor["name"], color="#e65c00", emoji="⚠️",
                severity="HIGH", category="Running / Panic Behavior",
                alert_type="RUNNING_PANIC", track_id=track_id, context=context,
                reasoning=reasoning,
                action="Investigate root cause. Update safety log. "
                       "Check for environmental triggers (alarms, altercation, evacuation).",
                today_stats=today_stats,
            )
            ok = self._send_email(supervisor["email"],
                f"[HSE REPORT] HIGH Alert — Running/Panic | ID #{track_id} | {context['timestamp']}",
                supervisor_html)
            if ok:
                logger.info("[PanicResponseAgent] 📧 Supervisor escalated → %s", supervisor["email"])

        _trigger_buzzer("RUNNING_PANIC")


# ─────────────────────────────────────────────────────────
#  ESCALATION AGENT
#  Runs on a background thread, monitors memory for patterns
#  and auto-escalates without waiting for a new detection event
# ─────────────────────────────────────────────────────────

class EscalationAgent(BaseAgent):
    """
    Background autonomous agent.
    Continuously monitors incident memory.
    Auto-escalates when repeat thresholds are breached.
    """

    def __init__(self, email_cfg: dict, memory: IncidentMemory, csv_log_path: str):
        super().__init__(email_cfg)
        self._memory = memory
        self._csv_log_path = csv_log_path
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()
        logger.info("[EscalationAgent] 🔍 Background monitoring started")

    def stop(self):
        self._running = False

    def _monitor_loop(self):
        """Runs every 60 seconds. Checks for weekly trend spikes."""
        while self._running:
            time.sleep(60)
            self._check_weekly_trend()

    def _check_weekly_trend(self):
        """
        If today's incident count exceeds the 7-day daily average by 2x,
        auto-send a trend alert to supervisor.
        """
        weekly = self._memory.get_historical_summary(days=7)
        today_str = datetime.date.today().isoformat()
        today_count = weekly["by_day"].get(today_str, 0)

        if not weekly["by_day"] or len(weekly["by_day"]) < 2:
            return  # Not enough data yet

        # Average excluding today
        other_days = {k: v for k, v in weekly["by_day"].items() if k != today_str}
        if not other_days:
            return
        avg = sum(other_days.values()) / len(other_days)

        escalation_key = f"trend_spike_{today_str}"
        if today_count >= max(avg * 2, 5) and not self._memory.is_escalated(escalation_key):
            self._memory.mark_escalated(escalation_key)
            self._send_trend_alert(today_count, avg, weekly)

    def _send_trend_alert(self, today_count: int, avg: float, weekly: dict):
        supervisor = self._cfg["supervisor"]
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%B %d, %Y")

        rows = "".join(
            f'<tr style="background:{"#f9f9f9" if i%2==0 else "#fff"};">'
            f'<td style="padding:6px 10px;">{d}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{c}</td></tr>'
            for i, (d, c) in enumerate(sorted(weekly["by_day"].items()))
        )

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
          <div style="max-width:600px;margin:auto;background:#fff;border-radius:8px;overflow:hidden;
                      box-shadow:0 2px 8px rgba(0,0,0,0.12);">
            <div style="background:#6a0dad;padding:20px 24px;">
              <h2 style="color:#fff;margin:0;font-size:18px;">📊 HSE AI — Trend Spike Alert</h2>
              <p style="color:#ddd;margin:6px 0 0;font-size:12px;">
                Auto-generated by EscalationAgent · {timestamp}
              </p>
            </div>
            <div style="padding:24px;">
              <p style="font-size:14px;color:#333;">Dear <strong>{supervisor['name']}</strong>,</p>
              <p style="font-size:14px;color:#555;">
                The HSE EscalationAgent has autonomously detected an <strong>abnormal incident
                frequency spike</strong> today. Today's count of
                <strong style="color:#cc0000;">{today_count} incidents</strong> is significantly
                above the 7-day daily average of <strong>{avg:.1f}</strong>.
              </p>
              <div style="background:#fff3e0;border-left:4px solid #ff9800;padding:14px 18px;
                          border-radius:4px;margin:16px 0;font-size:13px;color:#555;">
                🤖 <strong>Agent Reasoning:</strong> A 2× spike above baseline may indicate
                an unreported environmental hazard, equipment failure, or unsafe work practice
                that has not yet been identified. Immediate site review is recommended.
              </div>
              <h3 style="font-size:13px;color:#333;margin-bottom:8px;">7-Day Incident Log</h3>
              <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr style="background:#6a0dad;color:#fff;">
                  <th style="padding:8px 10px;text-align:left;">Date</th>
                  <th style="padding:8px 10px;">Incidents</th>
                </tr>
                {rows}
              </table>
              <p style="font-size:12px;color:#999;margin-top:20px;">
                Worst day this week: <strong>{weekly['worst_day']}</strong> ·
                Total 7-day incidents: <strong>{weekly['total']}</strong>
              </p>
            </div>
            <div style="background:#6a0dad;padding:10px 24px;text-align:center;">
              <p style="font-size:11px;color:#ddd;margin:0;">
                HSE AI EscalationAgent · {date_str}
              </p>
            </div>
          </div>
        </body></html>
        """
        ok = self._send_email(supervisor["email"],
            f"[HSE TREND ALERT] Incident Spike Detected — {today_count} incidents today | {timestamp}",
            html)
        if ok:
            logger.info("[EscalationAgent] 📊 Trend spike alert sent → %s", supervisor["email"])


# ─────────────────────────────────────────────────────────
#  HSE ORCHESTRATOR — coordinates all sub-agents
# ─────────────────────────────────────────────────────────

class HSEAgent:
    """
    Multi-agent orchestrator.

    On each incident:
      1. Records into shared IncidentMemory
      2. Routes to the correct specialist sub-agent
      3. EscalationAgent runs independently in background

    No LLM. No external API. Full agentic behavior through
    autonomous routing, memory-driven decisions, and background monitoring.
    """

    def __init__(self, email_cfg: dict, csv_log_path: str = "alerts/logs.csv"):
        self._cfg = email_cfg
        self._cooldown = email_cfg.get("cooldown_seconds", 30)
        self._last_action: dict[str, float] = {}

        # Shared memory across all agents
        self._memory = IncidentMemory(csv_log_path)

        # Specialist sub-agents
        self._fall_agent = FallResponseAgent(email_cfg)
        self._panic_agent = PanicResponseAgent(email_cfg)

        # Background escalation agent — starts monitoring immediately
        self._escalation_agent = EscalationAgent(email_cfg, self._memory, csv_log_path)
        self._escalation_agent.start()

        logger.info("[HSEOrchestrator] ✅ Multi-agent system initialized (3 agents active)")

    def execute_incident_protocol(self, alert_type: str, track_id: int):
        """
        Orchestrator entry point. Routes incident to specialist agent.
        """
        incident_key = f"{alert_type}_{track_id}"
        now = time.time()

        # Cooldown check
        if now - self._last_action.get(incident_key, 0) < self._cooldown:
            return
        self._last_action[incident_key] = now

        # Record into shared memory before routing
        self._memory.record(alert_type, track_id)

        # Build context
        context = _build_context()

        logger.info(
            "[HSEOrchestrator] 🤖 Routing %s | ID: %s | Shift: %s",
            alert_type, track_id, context["shift"]
        )

        # Route to specialist agent
        if alert_type == "SLIP_FALL":
            self._fall_agent.respond(track_id, self._memory, context)
        elif alert_type == "RUNNING_PANIC":
            self._panic_agent.respond(track_id, self._memory, context)
        else:
            logger.warning("[HSEOrchestrator] Unknown alert type: %s", alert_type)


# ─────────────────────────────────────────────────────────
#  SHARED UTILITIES — context builder, buzzer, email templates
# ─────────────────────────────────────────────────────────

def _build_context() -> dict:
    now = datetime.datetime.now()
    hour = now.hour
    if 9 <= hour < 18:
        shift, shift_label = "OFFICE_HOURS", "Office Hours"
    elif 18 <= hour < 22:
        shift, shift_label = "EVENING_SHIFT", "Evening Shift"
    else:
        shift, shift_label = "NIGHT_SHIFT", "Night Shift"
    return {
        "shift": shift,
        "shift_label": shift_label,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date_str": now.strftime("%B %d, %Y"),
    }


def _trigger_buzzer(alert_type: str):
    logger.info("[IoT] ⚡ Buzzer triggered for: %s", alert_type)
    print(f"⚡ [TOOL: IoT BUZZER] Emergency hooter activated for {alert_type}!")


def _build_guard_email(name, color, emoji, severity, category,
                        track_id, context, action, repeat_note) -> str:
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
      <div style="max-width:560px;margin:auto;background:#fff;border-radius:8px;overflow:hidden;
                  box-shadow:0 2px 8px rgba(0,0,0,0.1);">
        <div style="background:{color};padding:20px 24px;">
          <h2 style="color:#fff;margin:0;font-size:20px;">{emoji} {severity} — {category}</h2>
          <p style="color:#ffe0e0;margin:6px 0 0;font-size:12px;">HSE AI · FallResponseAgent / PanicResponseAgent</p>
        </div>
        <div style="padding:24px;">
          <p style="font-size:15px;color:#333;">Dear <strong>{name}</strong>,</p>
          <div style="background:#fff8f8;border-left:4px solid {color};padding:14px 18px;
                      margin:16px 0;border-radius:4px;font-size:13px;color:#444;">
            <table style="width:100%;border-collapse:collapse;">
              <tr><td style="width:130px;padding:3px 0;"><strong>Incident</strong></td><td>{category}</td></tr>
              <tr><td style="padding:3px 0;"><strong>Severity</strong></td>
                  <td style="color:{color};font-weight:bold;">{severity}</td></tr>
              <tr><td style="padding:3px 0;"><strong>Person ID</strong></td><td>#{track_id}</td></tr>
              <tr><td style="padding:3px 0;"><strong>Time</strong></td><td>{context['timestamp']}</td></tr>
              <tr><td style="padding:3px 0;"><strong>Shift</strong></td><td>{context['shift_label']}</td></tr>
            </table>
          </div>
          <div style="background:#fff8e1;border:1px solid #ffc107;border-radius:6px;padding:12px 16px;
                      font-size:13px;color:#555;margin-bottom:12px;">
            🧠 <strong>Agent Note:</strong> {repeat_note}
          </div>
          <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px 16px;
                      font-size:13px;color:#555;">
            ⚡ <strong>Your Action:</strong> {action}
          </div>
          <p style="font-size:11px;color:#aaa;margin-top:18px;">
            Auto-generated by HSE Agentic AI · Do not reply
          </p>
        </div>
        <div style="background:#f0f0f0;padding:10px 24px;text-align:center;">
          <p style="font-size:11px;color:#aaa;margin:0;">HSE AI Safety Monitoring · {context['date_str']}</p>
        </div>
      </div>
    </body></html>"""


def _build_supervisor_email(name, color, emoji, severity, category,
                              alert_type, track_id, context,
                              reasoning, action, today_stats) -> str:
    by_person = today_stats.get("by_person", {})
    top_persons = sorted(by_person.items(), key=lambda x: x[1], reverse=True)[:5]
    person_rows = "".join(
        f'<tr style="background:{"#f9f9f9" if i%2==0 else "#fff"};">'
        f'<td style="padding:5px 10px;">Person ID #{pid}</td>'
        f'<td style="padding:5px 10px;text-align:center;">{cnt}</td></tr>'
        for i, (pid, cnt) in enumerate(top_persons)
    ) or '<tr><td colspan="2" style="padding:8px;color:#aaa;text-align:center;">No data</td></tr>'

    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
      <div style="max-width:600px;margin:auto;background:#fff;border-radius:8px;overflow:hidden;
                  box-shadow:0 2px 8px rgba(0,0,0,0.1);">
        <div style="background:#1a1a2e;padding:20px 24px;">
          <h2 style="color:#fff;margin:0;font-size:18px;">HSE AI — Incident Report</h2>
          <p style="color:#aaa;margin:6px 0 0;font-size:12px;">
            {emoji} {severity} · {category} · {context['timestamp']}
          </p>
        </div>
        <div style="background:{color};padding:8px 24px;">
          <p style="color:#fff;margin:0;font-size:13px;font-weight:bold;">
            {emoji} {severity}: {category} — Person ID #{track_id}
          </p>
        </div>
        <div style="padding:24px;">
          <p style="font-size:14px;color:#333;">Dear <strong>{name}</strong>,</p>
          <table style="font-size:13px;color:#444;width:100%;border-collapse:collapse;margin-bottom:16px;">
            <tr style="background:#f9f9f9;">
              <td style="padding:7px 10px;width:170px;font-weight:bold;">Alert Code</td>
              <td style="padding:7px 10px;">{alert_type}</td></tr>
            <tr><td style="padding:7px 10px;font-weight:bold;">Person Track ID</td>
                <td style="padding:7px 10px;">#{track_id}</td></tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:7px 10px;font-weight:bold;">Date &amp; Time</td>
              <td style="padding:7px 10px;">{context['timestamp']}</td></tr>
            <tr><td style="padding:7px 10px;font-weight:bold;">Shift</td>
                <td style="padding:7px 10px;">{context['shift_label']}</td></tr>
            <tr style="background:#f9f9f9;">
              <td style="padding:7px 10px;font-weight:bold;">Guard Notified</td>
              <td style="padding:7px 10px;">✅ Simultaneously</td></tr>
          </table>
          <div style="background:#eef2ff;border-left:4px solid #4361ee;padding:12px 16px;
                      border-radius:4px;font-size:13px;color:#333;margin-bottom:16px;">
            🤖 <strong>Agent Reasoning:</strong> {reasoning}
          </div>
          <div style="background:#f0fff4;border:1px solid #28a745;border-radius:6px;
                      padding:12px 16px;font-size:13px;color:#555;margin-bottom:20px;">
            📌 <strong>Required Action:</strong> {action}
          </div>
          <h3 style="font-size:13px;color:#333;margin-bottom:8px;">📊 Today's Incident Intelligence</h3>
          <p style="font-size:12px;color:#777;margin:0 0 8px;">
            Total today: <strong>{today_stats['total']}</strong> ·
            Peak hour: <strong>{today_stats['peak_hour']}</strong>
          </p>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:#1a1a2e;color:#fff;">
              <th style="padding:7px 10px;text-align:left;">Person</th>
              <th style="padding:7px 10px;">Incidents Today</th>
            </tr>
            {person_rows}
          </table>
          <p style="font-size:11px;color:#aaa;margin-top:18px;">
            Auto-generated by HSE Agentic AI · Screenshots saved locally
          </p>
        </div>
        <div style="background:#1a1a2e;padding:10px 24px;text-align:center;">
          <p style="font-size:11px;color:#555;margin:0;">
            HSE Multi-Agent System · {context['date_str']}
          </p>
        </div>
      </div>
    </body></html>"""
