import os
import sys
import tempfile
import logging

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume
from evaluator import ResumeEvaluator
from transform import convert_json_resume_to_text, convert_github_data_to_text
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from config import DEVELOPMENT_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REACT_BUILD = os.path.join(os.path.dirname(__file__), "frontend", "dist")

app = Flask(__name__, static_folder=REACT_BUILD, static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

CATEGORY_MAXES = {
    "open_source": 35,
    "self_projects": 30,
    "production": 25,
    "technical_skills": 10,
}


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )



@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path and os.path.exists(os.path.join(REACT_BUILD, path)):
        return send_from_directory(REACT_BUILD, path)
    
    # Fallback to templates/index.html if frontend is not built
    if not os.path.exists(REACT_BUILD) or not path:
        return render_template("index.html")
        
    return send_from_directory(REACT_BUILD, "index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # 1. Parse PDF
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(tmp_path)

        if resume_data is None:
            return jsonify({"error": "Failed to parse the resume PDF"}), 500

        # 2. Fetch GitHub data
        github_data = {}
        profiles = []
        if resume_data.basics:
            profiles = resume_data.basics.profiles or []
        github_profile = find_profile(profiles, "Github")
        if github_profile:
            try:
                github_data = fetch_and_display_github_info(github_profile.url)
            except Exception as e:
                logger.warning(f"GitHub fetch failed: {e}")

        # 3. Evaluate
        resume_text = convert_json_resume_to_text(resume_data)
        if github_data:
            resume_text += convert_github_data_to_text(github_data)

        model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
        evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)
        evaluation = evaluator.evaluate_resume(resume_text)

        if evaluation is None:
            return jsonify({"error": "Evaluation failed — please try again"}), 500

        # 4. Build response
        total_score = 0.0
        max_score = 0
        scores = {}
        for cat, cat_data in evaluation.scores.model_dump().items():
            capped = min(cat_data["score"], CATEGORY_MAXES.get(cat, cat_data["max"]))
            total_score += capped
            max_score += cat_data["max"]
            scores[cat] = {
                "score": capped,
                "max": cat_data["max"],
                "evidence": cat_data["evidence"],
            }

        total_score += evaluation.bonus_points.total
        total_score -= evaluation.deductions.total
        total_score = round(total_score, 1)

        candidate = {}
        if resume_data.basics:
            b = resume_data.basics
            loc_parts = [b.location.city, b.location.countryCode] if b.location else []
            candidate = {
                "name": b.name,
                "email": b.email,
                "location": ", ".join(p for p in loc_parts if p) or None,
                "summary": b.summary,
            }

        return jsonify(
            {
                "candidate": candidate,
                "total_score": total_score,
                "max_score": max_score,
                "scores": scores,
                "bonus_points": {
                    "total": evaluation.bonus_points.total,
                    "breakdown": evaluation.bonus_points.breakdown,
                },
                "deductions": {
                    "total": evaluation.deductions.total,
                    "reasons": evaluation.deductions.reasons,
                },
                "key_strengths": evaluation.key_strengths,
                "areas_for_improvement": evaluation.areas_for_improvement,
            }
        )

    except Exception as e:
        logger.exception("Evaluation error")
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
