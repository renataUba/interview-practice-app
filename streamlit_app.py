import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

st.title("Interview Practice with Gemini")

# ── Response Schema ────────────────────────────────────────────────────────────

class TechnicalSkill(BaseModel):
    skill: str
    example_interview_question: str

class JobAnalysis(BaseModel):
    job_title: str
    location: str
    work_mode: str
    language_requirements: list[str]
    years_of_experience: int
    technical_skills: list[TechnicalSkill]  # exactly 5
    soft_skills: list[str]                  # exactly 2

# ── Gemini Setup ───────────────────────────────────────────────────────────────

load_dotenv("geminai.env")
api_key = os.getenv("Default_Gemini_API_Key_Free")
if not api_key:
    raise ValueError("Please set your API key in geminai.env or environment")

client = genai.Client(api_key=api_key)

# ── Session State ──────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "technical_skills" not in st.session_state:
    st.session_state.technical_skills = []
if "clicked_skill" not in st.session_state:
    st.session_state.clicked_skill = None
if "skill_resource" not in st.session_state:
    st.session_state.skill_resource = {}

# ── URL Input ──────────────────────────────────────────────────────────────────

url = st.text_input("Enter your Job ad URL:")

if st.button("Consult Gemini"):
    if not url:
        st.warning("Please provide job ad URL.")
    else:
        try:
            # Fetch and clean HTML
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.extract()

            page_text = soup.get_text(separator=" ", strip=True)
            MAX_CHARS = 15000
            if len(page_text) > MAX_CHARS:
                page_text = page_text[:MAX_CHARS]
                st.caption("Page content was trimmed to fit analysis limits.")

            # Prompt
            SYSTEM_INSTRUCTION = """
            You are an expert recruiter specialised in IT roles.
            You only answer questions related to interview preparation.
            Refuse: illegal content, hacking, medical/legal advice, explicit content.
            """.strip()

            OUTPUT_INSTRUCTIONS = """
            Analyse the job ad and return structured JSON with:
            1. Job title, location, work mode (remote/on-site/hybrid), language requirements, years of experience.
            2. Exactly 5 technical skills ranked by importance, each with one realistic interview question.
            3. Exactly 2 soft skills most critical for the role.
            Return ONLY valid JSON. No markdown, no explanation.
            Output should be in English language.
            """.strip()

            prompt = f"{OUTPUT_INSTRUCTIONS}\n\n--- JOB AD ---\n{page_text}\n--- END ---"

            # Gemini call
            gemini_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=40,
                    response_mime_type="application/json",
                    response_schema=JobAnalysis,
                ),
            )

            # Parse and store in session state
            result = JobAnalysis.model_validate_json(gemini_response.text)
            st.session_state.result = result
            st.session_state.technical_skills = [ts.skill for ts in result.technical_skills]
            st.session_state.clicked_skill = None   # reset on new search
            st.session_state.skill_resource = {}    # reset on new search

        except requests.exceptions.HTTPError as e:
            st.error(f"Failed to fetch the page (HTTP error): {e}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the URL. Check the address and try again.")
        except requests.exceptions.InvalidURL:
            st.error("The URL appears to be invalid. Make sure it starts with https://")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Display Job Analysis ───────────────────────────────────────────────────────

if st.session_state.result:
    result = st.session_state.result

    st.subheader("Job Overview")
    st.write(f"**Role:** {result.job_title} — {result.location} ({result.work_mode})")
    st.write(f"**Experience:** {result.years_of_experience} years")
    st.write(f"**Languages:** {', '.join(result.language_requirements)}")

    st.subheader("Click on a technical skill to see a practice question")
    for i, ts in enumerate(result.technical_skills, 1):
        with st.expander(f"{i}. {ts.skill}"):
            st.write(f"**Interview Q:** {ts.example_interview_question}")

    st.subheader("Soft Skills")
    for ss in result.soft_skills:
        st.write(f"• {ss}")

# ── Clickable Skill Buttons ────────────────────────────────────────────────────

if st.session_state.technical_skills:
    st.subheader("Click on a skill to find a learning resource:")
    for skill in st.session_state.technical_skills:
        if st.button(skill):
            st.session_state.clicked_skill = skill

# ── Fetch Gemini Learning Resource ────────────────────────────────────────────

if st.session_state.clicked_skill:
    skill = st.session_state.clicked_skill

    if skill not in st.session_state.skill_resource:
        resource_prompt = f"""
        ZERO-SHOT + role Prompt:
        You are a senior technical mentor.
        Provide one link to a documentation, tutorial, or course where someone can learn: {skill}.
        Only use well-known, stable domains (official docs, major learning platforms, universities).
        If uncertain about a URL, provide the platform homepage and exact course name instead.
        Prefer: Coursera, edX, Udemy, freeCodeCamp, MIT OpenCourseWare, Stanford Online, Harvard Online.
        Return ONLY valid JSON:
        {{
            "skill": "{skill}",
            "resource": "URL to learn this skill"
        }}
        """.strip()

        try:
            gemini_resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=resource_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.9,
                    top_k=40,
                    response_mime_type="application/json",
                ),
            )
            import json
            data = json.loads(gemini_resp.text)
            st.session_state.skill_resource[skill] = data.get("resource", None)
        except Exception as e:
            st.error(f"Error fetching resource: {e}")
            st.session_state.skill_resource[skill] = None

# ── Display Learning Resource ──────────────────────────────────────────────────

if st.session_state.clicked_skill:
    skill = st.session_state.clicked_skill
    resource = st.session_state.skill_resource.get(skill)
    if resource:
        st.markdown(
            f'<a href="{resource}" target="_blank" rel="noopener noreferrer">'
            f'Learn {skill}</a>',
            unsafe_allow_html=True,
        )


# different prompting techniques:

        #  #FEW-SHOT Prompt:
        # You are a senior technical mentor.
        # Provide one link to a documentation, tutorial, or course where someone can learn: {skill}.
        # Only use well-known, stable domains (official docs, major learning platforms, universities).
        # If uncertain about a URL, provide the platform homepage and exact course name instead.
        # Prefer: Coursera, edX, Udemy, freeCodeCamp, MIT OpenCourseWare, Stanford Online, Harvard Online.

        # Here are examples of the expected output format:

        # Example 1:
        # Skill: Machine Learning
        # Output: {{"skill": "Machine Learning", "resource": "https://www.coursera.org/learn/machine-learning"}}

        # Example 2:
        # Skill: Web Development
        # Output: {{"skill": "Web Development", "resource": "https://www.freecodecamp.org/learn/responsive-web-design/"}}

        # Example 3:
        # Skill: Data Structures and Algorithms
        # Output: {{"skill": "Data Structures and Algorithms", "resource": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"}}

        # Now, return ONLY valid JSON for the following skill:
        # {{
        #     "skill": "{skill}",
        #     "resource": "URL to learn this skill"
        # }}

        #chain of thought
        # You are a senior technical mentor.
        # Provide one link to a documentation, tutorial, or course where someone can learn: {skill}.
        # Only use well-known, stable domains (official docs, major learning platforms, universities).
        # If uncertain about a URL, provide the platform homepage and exact course name instead.
        # Prefer: Coursera, edX, Udemy, freeCodeCamp, MIT OpenCourseWare, Stanford Online, Harvard Online.

        # Before providing your answer, think through the following steps:

        # Step 1 - Classify the skill type:
        # Ask yourself: Is this a programming language, a framework, a concept, a tool, or a soft skill?
        # This determines whether official docs or a structured course is more appropriate.

        # Step 2 - Identify the best resource type:
        # - Programming languages → Official documentation (e.g., docs.python.org)
        # - Frameworks/Libraries → Official docs or freeCodeCamp
        # - Academic concepts (ML, AI, Algorithms) → MIT OCW, Stanford Online, Coursera
        # - General tech skills → Coursera, edX, freeCodeCamp, Udemy

        # Step 3 - Select a specific resource:
        # Name the platform you've chosen and why it's the best fit for this skill.

        # Step 4 - Verify URL confidence:
        # Ask yourself: Am I confident this exact URL exists and is stable?
        # - If YES → use the full URL
        # - If NO → use the platform homepage + exact course/page name

        # Step 5 - Format the output:
        # Return ONLY valid JSON with no explanation outside it:
        # {{
        #     "skill": "{skill}",
        #     "resource": "URL to learn this skill"
        # }}

        # Example of this reasoning process:
        # Skill: Docker
        # Step 1: Docker is a tool/platform.
        # Step 2: Tools are best learned via official docs or structured tutorials.
        # Step 3: docs.docker.com is the official, stable, well-maintained documentation.
        # Step 4: I am confident https://docs.docker.com/get-started/ is a valid, stable URL.
        # Step 5: {{"skill": "Docker", "resource": "https://docs.docker.com/get-started/"}}

        # Now apply this reasoning to: {skill}
        # Return ONLY the final JSON output.

        #Self-Consistency:
        # You are a senior technical mentor. Provide one resource link to learn: {skill}.
        # Only use well-known, stable domains. Prefer: Coursera, edX, freeCodeCamp, MIT OCW, Stanford Online, official docs.

        # Generate 3 independent reasoning paths, then pick the most consistent answer:

        # Path 1 - Official source angle:
        # What is the most authoritative/official resource for {skill}?

        # Path 2 - Learner angle:
        # What resource would best suit a complete beginner learning {skill} from scratch?

        # Path 3 - Popularity angle:
        # What is the most widely recommended resource for {skill} across the developer community?

        # Final decision:
        # Compare the 3 paths. If 2 or more agree on the same resource → use it.
        # If all 3 differ → default to the most authoritative source (Path 1).
        # If uncertain about any URL → use platform homepage + exact course name.

        # Return ONLY valid JSON:
        # {{
        #     "skill": "{skill}",
        #     "resource": "URL to learn this skill"
        # }}