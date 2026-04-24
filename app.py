"""
GigOptimizer AI - Backend
Uses Z.AI's GLM (ilmu-glm-5.1) for intelligent gig worker optimization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import json
import random
from datetime import datetime, timedelta
import math

load_dotenv()

app = Flask(__name__)
CORS(app)

# Z.AI GLM Configuration
GLM_URL = "https://api.ilmu.ai/v1/chat/completions"
GLM_MODEL = "ilmu-glm-5.1"
API_KEY = os.getenv("API_KEY", "demo-key")

# ─────────────────────────────────────────────
# GLM Core Interface
# ─────────────────────────────────────────────

def call_glm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    """
    Central GLM call — all intelligence flows through here.
    If this is removed, the system cannot generate insights or decisions.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }

    try:
        response = requests.post(GLM_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            # Fallback to demo mode for showcase if API not configured
            return {"success": False, "error": f"GLM API error {response.status_code}", "demo": True}
    except Exception as e:
        return {"success": False, "error": str(e), "demo": True}


def glm_extract_text(glm_result: dict) -> str:
    """Extract text content from GLM response."""
    try:
        return glm_result["data"]["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ─────────────────────────────────────────────
# Simulated Gig Task Data
# ─────────────────────────────────────────────

SAMPLE_TASKS = [
    {"id": "T001", "title": "Deliver groceries - Taman Jaya", "type": "delivery", "distance_km": 3.2, "estimated_minutes": 25, "payout_myr": 12.50, "surge_multiplier": 1.0, "rating_requirement": 4.2, "time_window": "09:00-11:00", "weather": "clear", "traffic": "low"},
    {"id": "T002", "title": "Food pickup - Midvalley to Bangsar", "type": "food_delivery", "distance_km": 5.8, "estimated_minutes": 40, "payout_myr": 18.00, "surge_multiplier": 1.5, "rating_requirement": 4.0, "time_window": "12:00-13:30", "weather": "clear", "traffic": "high"},
    {"id": "T003", "title": "Freelance logo design - 3 revisions", "type": "freelance_design", "distance_km": 0, "estimated_minutes": 180, "payout_myr": 120.00, "surge_multiplier": 1.0, "rating_requirement": 4.5, "time_window": "flexible", "weather": "n/a", "traffic": "n/a"},
    {"id": "T004", "title": "Parcel delivery - PJ to KL Sentral", "type": "delivery", "distance_km": 12.1, "estimated_minutes": 55, "payout_myr": 22.00, "surge_multiplier": 1.2, "rating_requirement": 4.3, "time_window": "14:00-16:00", "weather": "rain", "traffic": "medium"},
    {"id": "T005", "title": "Ride-hail - Subang to KLCC", "type": "ride_hail", "distance_km": 18.4, "estimated_minutes": 50, "payout_myr": 28.50, "surge_multiplier": 2.0, "rating_requirement": 4.6, "time_window": "17:30-19:00", "weather": "clear", "traffic": "high"},
    {"id": "T006", "title": "Data entry task - 200 rows", "type": "freelance_data", "distance_km": 0, "estimated_minutes": 90, "payout_myr": 35.00, "surge_multiplier": 1.0, "rating_requirement": 4.0, "time_window": "flexible", "weather": "n/a", "traffic": "n/a"},
    {"id": "T007", "title": "Grocery delivery - Damansara", "type": "delivery", "distance_km": 2.5, "estimated_minutes": 20, "payout_myr": 10.00, "surge_multiplier": 1.0, "rating_requirement": 4.1, "time_window": "10:00-11:00", "weather": "rain", "traffic": "low"},
    {"id": "T008", "title": "Evening surge ride - Bukit Bintang", "type": "ride_hail", "distance_km": 8.3, "estimated_minutes": 30, "payout_myr": 24.00, "surge_multiplier": 1.8, "rating_requirement": 4.4, "time_window": "20:00-22:00", "weather": "clear", "traffic": "medium"},
]

WORKER_PROFILES = {
    "W001": {"name": "Ahmad Razif", "rating": 4.7, "vehicle": "motorcycle", "weekly_earnings": [420, 380, 510, 445, 390, 520, 480], "completed_tasks": 312, "preferred_types": ["delivery", "food_delivery"], "peak_hours": "morning", "fuel_cost_per_km": 0.18},
    "W002": {"name": "Siti Norlela", "rating": 4.5, "vehicle": "car", "weekly_earnings": [850, 920, 780, 1100, 950, 880, 1020], "completed_tasks": 187, "preferred_types": ["ride_hail", "delivery"], "peak_hours": "evening", "fuel_cost_per_km": 0.35},
    "W003": {"name": "Rajesh Kumar", "rating": 4.8, "vehicle": "bicycle", "weekly_earnings": [280, 310, 260, 340, 295, 320, 300], "completed_tasks": 89, "preferred_types": ["freelance_design", "freelance_data"], "peak_hours": "flexible", "fuel_cost_per_km": 0.0},
}


# ─────────────────────────────────────────────
# Helper: Compute metrics without GLM
# ─────────────────────────────────────────────

def compute_task_metrics(task: dict, worker: dict) -> dict:
    """Pure math — no intelligence, just numbers."""
    fuel_cost = task["distance_km"] * worker["fuel_cost_per_km"]
    net_pay = (task["payout_myr"] * task["surge_multiplier"]) - fuel_cost
    hourly_rate = (net_pay / task["estimated_minutes"]) * 60 if task["estimated_minutes"] > 0 else net_pay
    efficiency_score = (net_pay / max(task["estimated_minutes"], 1)) * 10
    return {
        "fuel_cost": round(fuel_cost, 2),
        "gross_pay": round(task["payout_myr"] * task["surge_multiplier"], 2),
        "net_pay": round(net_pay, 2),
        "hourly_rate": round(hourly_rate, 2),
        "efficiency_score": round(efficiency_score, 2),
    }


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": GLM_MODEL, "timestamp": datetime.now().isoformat()})


@app.route("/api/worker/<worker_id>", methods=["GET"])
def get_worker(worker_id):
    worker = WORKER_PROFILES.get(worker_id)
    if not worker:
        return jsonify({"error": "Worker not found"}), 404

    avg_earnings = sum(worker["weekly_earnings"]) / len(worker["weekly_earnings"])
    trend = worker["weekly_earnings"][-1] - worker["weekly_earnings"][-2]

    return jsonify({
        "worker": worker,
        "stats": {
            "avg_weekly_earnings": round(avg_earnings, 2),
            "earnings_trend": round(trend, 2),
            "trend_direction": "up" if trend > 0 else "down"
        }
    })


@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """Return available tasks with pre-computed metrics."""
    worker_id = request.args.get("worker_id", "W001")
    worker = WORKER_PROFILES.get(worker_id, WORKER_PROFILES["W001"])

    enriched = []
    for task in SAMPLE_TASKS:
        metrics = compute_task_metrics(task, worker)
        enriched.append({**task, "metrics": metrics})

    return jsonify({"tasks": enriched, "count": len(enriched)})


@app.route("/api/analyze/tasks", methods=["POST"])
def analyze_tasks():
    """
    GLM-powered task analysis.
    The GLM interprets structured task data + worker context to reason about
    which tasks are best, why, and what trade-offs exist.
    """
    body = request.json or {}
    worker_id = body.get("worker_id", "W001")
    worker = WORKER_PROFILES.get(worker_id, WORKER_PROFILES["W001"])

    # Compute raw metrics (structured data)
    task_metrics = []
    for task in SAMPLE_TASKS:
        m = compute_task_metrics(task, worker)
        task_metrics.append({**task, **m})

    # Build rich context for GLM (structured + unstructured fusion)
    context_block = f"""
WORKER PROFILE:
- Name: {worker['name']}
- Rating: {worker['rating']}/5.0
- Vehicle: {worker['vehicle']}
- Preferred task types: {', '.join(worker['preferred_types'])}
- Peak productivity hours: {worker['peak_hours']}
- Fuel cost: MYR {worker['fuel_cost_per_km']}/km
- Recent 7-day earnings (MYR): {worker['weekly_earnings']}
- Total completed tasks: {worker['completed_tasks']}

AVAILABLE TASKS (with computed metrics):
{json.dumps(task_metrics, indent=2)}

CURRENT CONDITIONS:
- Time of day: {datetime.now().strftime('%H:%M')} (Malaysia Time)
- Day: {datetime.now().strftime('%A')}
"""

    system_prompt = """You are GigOptimizer AI, an intelligent decision-support system for gig economy workers in Malaysia.

Your role is to analyze both STRUCTURED data (task metrics, earnings, distances) and UNSTRUCTURED signals (weather, traffic, surge patterns, worker preferences) to generate context-aware, actionable insights.

You must:
1. Recommend the TOP 3 tasks with clear reasoning
2. Identify hidden trade-offs (e.g., high pay but bad weather, surge but heavy traffic)
3. Provide a scheduling strategy for the day
4. Quantify the expected income impact (MYR)
5. Explain decisions in plain language a gig worker understands
6. Flag risks (weather, low efficiency, rating impact)

Respond in valid JSON with this exact structure:
{
  "top_recommendations": [
    {
      "task_id": "...",
      "rank": 1,
      "reason": "...",
      "expected_net_myr": ...,
      "risk_level": "low|medium|high",
      "risk_factors": ["..."]
    }
  ],
  "schedule_strategy": "...",
  "income_projection": {
    "if_follow_recommendations_myr": ...,
    "vs_random_selection_myr": ...,
    "improvement_percent": ...
  },
  "key_insight": "...",
  "avoid_tasks": ["task_id"],
  "avoid_reason": "..."
}"""

    glm_result = call_glm(system_prompt, f"Analyze and optimize task selection for this gig worker:\n{context_block}")

    if glm_result["success"]:
        raw_text = glm_extract_text(glm_result)
        try:
            # Strip markdown fences if present
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            analysis = json.loads(clean)
        except Exception:
            analysis = {"raw_insight": raw_text, "parse_error": True}
    else:
        # Demo fallback with realistic simulated GLM output
        analysis = _demo_task_analysis(task_metrics, worker)

    return jsonify({
        "worker_id": worker_id,
        "worker_name": worker["name"],
        "analysis": analysis,
        "glm_powered": glm_result["success"],
        "model": GLM_MODEL
    })


@app.route("/api/analyze/schedule", methods=["POST"])
def analyze_schedule():
    """
    GLM generates an optimized daily schedule.
    Reasons about time windows, task clustering, energy management.
    """
    body = request.json or {}
    worker_id = body.get("worker_id", "W001")
    worker = WORKER_PROFILES.get(worker_id, WORKER_PROFILES["W001"])
    target_earnings = body.get("target_earnings", 150)

    system_prompt = """You are a scheduling intelligence for gig workers. Generate an optimized daily schedule.

Consider:
- Task clustering by geography to reduce travel time
- Surge pricing windows (typically 7-9am, 12-2pm, 5-8pm)
- Worker energy patterns and vehicle constraints
- Weather impacts on outdoor tasks
- Income targets

Respond ONLY in valid JSON:
{
  "schedule": [
    {
      "time_slot": "HH:MM - HH:MM",
      "task_id": "...",
      "task_title": "...",
      "action": "...",
      "expected_earning_myr": ...
    }
  ],
  "total_projected_myr": ...,
  "total_hours": ...,
  "effective_hourly_rate": ...,
  "breaks": ["HH:MM - HH:MM"],
  "optimization_notes": "...",
  "time_saved_vs_unoptimized_minutes": ...
}"""

    user_prompt = f"""
Worker: {worker['name']} | Vehicle: {worker['vehicle']} | Peak hours: {worker['peak_hours']}
Daily earnings target: MYR {target_earnings}
Preferred task types: {worker['preferred_types']}

Available tasks:
{json.dumps(SAMPLE_TASKS, indent=2)}

Create an optimal schedule for today that maximizes earnings while respecting time windows and worker preferences.
"""

    glm_result = call_glm(system_prompt, user_prompt, temperature=0.5)

    if glm_result["success"]:
        raw_text = glm_extract_text(glm_result)
        try:
            clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            schedule = json.loads(clean)
        except Exception:
            schedule = {"raw": raw_text}
    else:
        schedule = _demo_schedule(worker, target_earnings)

    return jsonify({
        "worker_id": worker_id,
        "target_earnings_myr": target_earnings,
        "schedule": schedule,
        "glm_powered": glm_result["success"],
        "model": GLM_MODEL
    })


@app.route("/api/analyze/income", methods=["POST"])
def analyze_income():
    """
    GLM forecasts income trends and identifies revenue improvement strategies.
    Interprets historical earnings + market signals.
    """
    body = request.json or {}
    worker_id = body.get("worker_id", "W001")
    worker = WORKER_PROFILES.get(worker_id, WORKER_PROFILES["W001"])

    avg = sum(worker["weekly_earnings"]) / len(worker["weekly_earnings"])
    volatility = math.sqrt(sum((x - avg) ** 2 for x in worker["weekly_earnings"]) / len(worker["weekly_earnings"]))

    system_prompt = """You are an income optimization analyst for gig economy workers in Malaysia.

Analyze earnings patterns and provide strategic revenue advice.

Respond ONLY in valid JSON:
{
  "trend_analysis": "...",
  "peak_performance_days": ["..."],
  "revenue_strategies": [
    {"strategy": "...", "estimated_uplift_myr_weekly": ..., "effort": "low|medium|high", "timeframe": "..."}
  ],
  "monthly_forecast_myr": ...,
  "income_stability_score": ...,
  "key_risks": ["..."],
  "actionable_tip": "...",
  "comparison_to_median_gig_worker": "..."
}"""

    user_prompt = f"""
Worker: {worker['name']} | Vehicle: {worker['vehicle']}
7-day earnings (MYR): {worker['weekly_earnings']}
Average: MYR {avg:.2f}/week | Volatility: ±MYR {volatility:.2f}
Completed tasks: {worker['completed_tasks']}
Current rating: {worker['rating']}/5.0
Preferred types: {worker['preferred_types']}

Analyze income pattern and recommend strategies to increase weekly earnings by at least 15%.
"""

    glm_result = call_glm(system_prompt, user_prompt, temperature=0.6)

    if glm_result["success"]:
        raw_text = glm_extract_text(glm_result)
        try:
            clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            income_analysis = json.loads(clean)
        except Exception:
            income_analysis = {"raw": raw_text}
    else:
        income_analysis = _demo_income_analysis(worker, avg, volatility)

    return jsonify({
        "worker_id": worker_id,
        "current_avg_weekly_myr": round(avg, 2),
        "earnings_history": worker["weekly_earnings"],
        "analysis": income_analysis,
        "glm_powered": glm_result["success"],
        "model": GLM_MODEL
    })


@app.route("/api/analyze/tradeoff", methods=["POST"])
def analyze_tradeoff():
    """
    GLM performs trade-off analysis between competing tasks.
    This is pure decision intelligence — not achievable without GLM.
    """
    body = request.json or {}
    worker_id = body.get("worker_id", "W001")
    task_ids = body.get("task_ids", ["T002", "T005"])
    worker = WORKER_PROFILES.get(worker_id, WORKER_PROFILES["W001"])

    selected_tasks = [t for t in SAMPLE_TASKS if t["id"] in task_ids]
    metrics = [compute_task_metrics(t, worker) for t in selected_tasks]
    tasks_with_metrics = [{**t, **m} for t, m in zip(selected_tasks, metrics)]

    system_prompt = """You are a trade-off analysis engine for gig workers. Compare tasks across multiple dimensions and recommend the best choice with nuanced reasoning.

Consider: earnings, time efficiency, risk (weather/traffic/rating), energy cost, opportunity cost, and strategic value (ratings, repeat customers).

Respond ONLY in valid JSON:
{
  "winner_task_id": "...",
  "winner_reason": "...",
  "comparison_matrix": {
    "task_id_1": {"earnings_score": 0-10, "efficiency_score": 0-10, "risk_score": 0-10, "strategic_score": 0-10},
    "task_id_2": {"earnings_score": 0-10, "efficiency_score": 0-10, "risk_score": 0-10, "strategic_score": 0-10}
  },
  "hidden_factors": ["..."],
  "under_which_conditions_prefer_other": "...",
  "confidence_level": "low|medium|high",
  "decision_explanation": "..."
}"""

    user_prompt = f"""
Worker profile: {json.dumps(worker, indent=2)}

Compare these tasks and recommend which one to take:
{json.dumps(tasks_with_metrics, indent=2)}

Provide a nuanced decision with trade-off analysis.
"""

    glm_result = call_glm(system_prompt, user_prompt, temperature=0.4)

    if glm_result["success"]:
        raw_text = glm_extract_text(glm_result)
        try:
            clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            tradeoff = json.loads(clean)
        except Exception:
            tradeoff = {"raw": raw_text}
    else:
        tradeoff = _demo_tradeoff(tasks_with_metrics)

    return jsonify({
        "tasks_compared": tasks_with_metrics,
        "tradeoff_analysis": tradeoff,
        "glm_powered": glm_result["success"],
        "model": GLM_MODEL
    })


@app.route("/api/workers", methods=["GET"])
def list_workers():
    return jsonify({"workers": [{"id": k, "name": v["name"], "vehicle": v["vehicle"]} for k, v in WORKER_PROFILES.items()]})


# ─────────────────────────────────────────────
# Demo Fallbacks (realistic simulated GLM output)
# These demonstrate the SHAPE of GLM output for showcase
# ─────────────────────────────────────────────

def _demo_task_analysis(task_metrics, worker):
    return {
        "top_recommendations": [
            {"task_id": "T005", "rank": 1, "reason": "Evening surge at 2x multiplier aligns with your peak hours. Despite high traffic, the MYR 28.50 base becomes MYR 57.00 gross — your highest hourly rate today at MYR 68.40/hr net.", "expected_net_myr": 52.30, "risk_level": "medium", "risk_factors": ["Heavy traffic may extend journey by 15-20 min", "High rating requirement (4.6) — you qualify at 4.7"]},
            {"task_id": "T003", "rank": 2, "reason": "Freelance logo design has zero fuel cost and flexible timing. At MYR 120 for 3 hours, effective rate is MYR 40/hr. Can be batched with other tasks.", "expected_net_myr": 120.00, "risk_level": "low", "risk_factors": ["Time-intensive", "Requires design skills"]},
            {"task_id": "T002", "rank": 3, "reason": "Lunch surge (1.5x) at a route you know. MYR 27 gross after surge, despite high traffic the route is short enough to maintain profitability.", "expected_net_myr": 23.97, "risk_level": "medium", "risk_factors": ["High lunch traffic", "Tight time window 12:00-13:30"]}
        ],
        "schedule_strategy": "Start with T003 (flexible freelance) in the morning. Hit T002 during the 12-1:30pm lunch surge. Rest and position near Bukit Bintang by 5pm for T005 evening surge. This clusters high-value tasks around surge windows.",
        "income_projection": {"if_follow_recommendations_myr": 196.27, "vs_random_selection_myr": 142.50, "improvement_percent": 37.7},
        "key_insight": "Your motorcycle + morning preference means you're leaving 40% of income on the table by avoiding evening surge rides. T005 alone adds MYR 52 in one trip.",
        "avoid_tasks": ["T007"],
        "avoid_reason": "T007 pays MYR 10 but involves rain delivery on motorcycle — safety risk and poor MYR 30/hr effective rate. Not worth it today."
    }


def _demo_schedule(worker, target):
    return {
        "schedule": [
            {"time_slot": "08:00 - 11:00", "task_id": "T003", "task_title": "Freelance logo design", "action": "Complete from home before heading out", "expected_earning_myr": 120.00},
            {"time_slot": "11:30 - 11:50", "task_id": "T001", "task_title": "Deliver groceries - Taman Jaya", "action": "Quick nearby delivery on way out", "expected_earning_myr": 12.50},
            {"time_slot": "12:00 - 13:30", "task_id": "T002", "task_title": "Food pickup - Midvalley to Bangsar", "action": "Catch lunch surge window", "expected_earning_myr": 23.97},
            {"time_slot": "17:30 - 19:00", "task_id": "T005", "task_title": "Evening surge ride - Bukit Bintang", "action": "Position at Bukit Bintang 15 min early for 2x surge", "expected_earning_myr": 52.30},
        ],
        "total_projected_myr": 208.77,
        "total_hours": 7.5,
        "effective_hourly_rate": 27.84,
        "breaks": ["13:30 - 17:30"],
        "optimization_notes": "Tasks are geographically clustered. Mid-day break avoids low-demand afternoon slump. Evening surge adds 38% of daily income in 1.5 hours.",
        "time_saved_vs_unoptimized_minutes": 85
    }


def _demo_income_analysis(worker, avg, volatility):
    return {
        "trend_analysis": f"Your earnings show a positive trend (+MYR {worker['weekly_earnings'][-1] - worker['weekly_earnings'][0]:.0f} over 7 days) but with high volatility (±MYR {volatility:.0f}). Inconsistent task selection is the primary driver of income swings.",
        "peak_performance_days": ["Friday", "Saturday", "Monday morning"],
        "revenue_strategies": [
            {"strategy": "Target 2x+ surge windows exclusively for ride-hail tasks", "estimated_uplift_myr_weekly": 85, "effort": "low", "timeframe": "Immediate"},
            {"strategy": "Build freelance portfolio to increase design task rate from MYR 40/hr to MYR 65/hr", "estimated_uplift_myr_weekly": 120, "effort": "high", "timeframe": "4-6 weeks"},
            {"strategy": "Maintain 4.8+ rating to qualify for premium task pools", "estimated_uplift_myr_weekly": 45, "effort": "medium", "timeframe": "2 weeks"},
        ],
        "monthly_forecast_myr": round(avg * 4.3, 2),
        "income_stability_score": max(0, 10 - round(volatility / avg * 10, 1)),
        "key_risks": ["Over-reliance on a single task type", "Weather-dependent outdoor tasks in monsoon season", "Rating below 4.6 locks out premium tasks"],
        "actionable_tip": "Focus the next 14 days on surge-window ride-hails. This single change is projected to add MYR 340/month with zero additional hours worked.",
        "comparison_to_median_gig_worker": f"You earn MYR {avg:.0f}/week vs Malaysian gig worker median of MYR 380/week. You're in the top 35% — optimizing surge targeting could push you to top 15%."
    }


def _demo_tradeoff(tasks):
    t1, t2 = tasks[0], tasks[1]
    winner = t1 if t1.get("net_pay", 0) > t2.get("net_pay", 0) else t2
    loser = t2 if winner == t1 else t1
    return {
        "winner_task_id": winner["id"],
        "winner_reason": f"{winner['title']} offers MYR {winner.get('net_pay', 0):.2f} net vs MYR {loser.get('net_pay', 0):.2f} — {((winner.get('net_pay',1)/max(loser.get('net_pay',1),0.01))-1)*100:.0f}% better return. Surge multiplier ({winner.get('surge_multiplier',1)}x) is the key differentiator.",
        "comparison_matrix": {
            t1["id"]: {"earnings_score": min(10, round(t1.get("net_pay", 0)/8, 1)), "efficiency_score": min(10, round(t1.get("efficiency_score", 0), 1)), "risk_score": 7 if t1.get("weather") == "rain" else 8, "strategic_score": 7},
            t2["id"]: {"earnings_score": min(10, round(t2.get("net_pay", 0)/8, 1)), "efficiency_score": min(10, round(t2.get("efficiency_score", 0), 1)), "risk_score": 6 if t2.get("traffic") == "high" else 8, "strategic_score": 8}
        },
        "hidden_factors": ["Surge multiplier timing — T005 only available 17:30-19:00, missing it means waiting another day", "Rain on T007 increases accident risk disproportionately for motorcycle"],
        "under_which_conditions_prefer_other": f"If it's raining heavily and you're already near {loser['title'][:20]}, take the lower-value task — safety > earnings.",
        "confidence_level": "high",
        "decision_explanation": f"Based on net earnings, time efficiency, and current conditions, {winner['title']} is the clear winner. The {winner.get('surge_multiplier',1)}x surge more than compensates for any additional distance or time."
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
