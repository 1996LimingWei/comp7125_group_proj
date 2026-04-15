"""
Study Plan Feature
Manages study plan generation with user constraints and course information.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class WorkloadPreference(Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


@dataclass
class UserConstraints:
    """User constraints for study plan generation."""
    available_hours_per_week: Optional[int] = None
    goals: List[str] = field(default_factory=list)
    workload_preference: WorkloadPreference = WorkloadPreference.MODERATE
    preferred_days: List[str] = field(default_factory=lambda: [
                                      "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    course_codes: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_hours_per_week": self.available_hours_per_week,
            "goals": self.goals,
            "workload_preference": self.workload_preference.value,
            "preferred_days": self.preferred_days,
            "course_codes": self.course_codes,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@dataclass
class StudyPlan:
    """Generated study plan."""
    weekly_schedule: Dict[str, List[Dict[str, Any]]]
    total_hours_per_week: int
    recommendations: List[str]
    constraints_used: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weekly_schedule": self.weekly_schedule,
            "total_hours_per_week": self.total_hours_per_week,
            "recommendations": self.recommendations,
            "constraints_used": self.constraints_used,
        }


class StudyPlanManager:
    """
    Manages study plan generation workflow.
    Collects user constraints and generates personalized study plans.
    """

    def __init__(self, retrieval_service=None):
        """
        Initialize StudyPlanManager.

        Args:
            retrieval_service: Service for retrieving course information (lexical or neural)
        """
        self.retrieval_service = retrieval_service
        self.user_constraints: Optional[UserConstraints] = None
        # collecting_constraints, generating, complete
        self.conversation_state = "collecting_constraints"

    def start_study_plan_flow(self) -> str:
        """Start the study plan generation dialogue."""
        self.conversation_state = "collecting_constraints"
        self.user_constraints = UserConstraints()
        return (
            "I'll help you create a personalized study plan!\n\n"
            "To get started, please tell me:\n"
            "1. How many hours per week can you dedicate to studying?\n"
            "2. What are your main goals? (e.g., 'pass all courses', 'achieve high grades', 'balance work and study')\n"
            "3. What's your preferred workload? (light/moderate/heavy)\n"
            "4. Which days of the week work best for you?\n"
            "5. Any specific courses you want to focus on?"
        )

    def collect_constraint(self, user_input: str, constraint_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse user input to extract constraints.

        Args:
            user_input: User's response
            constraint_type: Type of constraint being collected (optional)

        Returns:
            Dict with status, message, and next action
        """
        if self.user_constraints is None:
            self.user_constraints = UserConstraints()

        # Try to parse various constraint types from input
        parsed = self._parse_constraints(user_input)

        # Update constraints
        if "hours" in parsed:
            self.user_constraints.available_hours_per_week = parsed["hours"]
        if "goals" in parsed:
            self.user_constraints.goals.extend(parsed["goals"])
        if "workload" in parsed:
            self.user_constraints.workload_preference = parsed["workload"]
        if "days" in parsed:
            self.user_constraints.preferred_days = parsed["days"]
        if "courses" in parsed:
            self.user_constraints.course_codes.extend(parsed["courses"])

        # Check if we have enough information
        missing = self._get_missing_constraints()

        if not missing:
            self.conversation_state = "ready_to_generate"
            return {
                "status": "ready",
                "message": self._format_constraints_summary(),
                "constraints": self.user_constraints.to_dict(),
                "next_action": "generate_plan",
            }
        else:
            return {
                "status": "collecting",
                "message": f"Thanks! I still need to know about: {', '.join(missing)}",
                "constraints": self.user_constraints.to_dict(),
                "next_action": "continue_collection",
            }

    def _parse_constraints(self, text: str) -> Dict[str, Any]:
        """Parse constraints from natural language input."""
        parsed = {}
        text_lower = text.lower()

        # Parse hours
        import re
        hour_patterns = [
            r'(\d+)\s*hours?\s*(per\s*week|weekly)?',
            r'(\d+)\s*h\s*(per\s*week|weekly)?',
            r'(\d+)\s*hrs?',
        ]
        for pattern in hour_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed["hours"] = int(match.group(1))
                break

        # Parse workload preference
        if any(word in text_lower for word in ["light", "easy", "relaxed"]):
            parsed["workload"] = WorkloadPreference.LIGHT
        elif any(word in text_lower for word in ["heavy", "intensive", "hard", "challenging"]):
            parsed["workload"] = WorkloadPreference.HEAVY
        elif any(word in text_lower for word in ["moderate", "medium", "balanced", "normal"]):
            parsed["workload"] = WorkloadPreference.MODERATE

        # Parse days
        days = ["monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday"]
        found_days = []
        for day in days:
            if day in text_lower:
                found_days.append(day.capitalize())
        if found_days:
            parsed["days"] = found_days

        # Parse course codes (e.g., COMP7125, DAAI)
        course_pattern = r'\b[A-Z]{2,4}\d{4}\b'
        courses = re.findall(course_pattern, text.upper())
        if courses:
            parsed["courses"] = courses

        # Parse goals (simple keyword extraction)
        goal_keywords = ["pass", "high grade", "distinction",
                         "balance", "work", "learn", "master"]
        goals = []
        for keyword in goal_keywords:
            if keyword in text_lower:
                goals.append(keyword)
        if goals:
            parsed["goals"] = goals

        return parsed

    def _get_missing_constraints(self) -> List[str]:
        """Check which constraints are still missing."""
        missing = []
        if self.user_constraints.available_hours_per_week is None:
            missing.append("available hours per week")
        if not self.user_constraints.goals:
            missing.append("study goals")
        if not self.user_constraints.preferred_days:
            missing.append("preferred days")
        return missing

    def _format_constraints_summary(self) -> str:
        """Format collected constraints for user confirmation."""
        c = self.user_constraints
        return (
            f"Great! Here's what I've collected:\n\n"
            f"📅 Available Hours: {c.available_hours_per_week} hours/week\n"
            f"🎯 Goals: {', '.join(c.goals) if c.goals else 'Not specified'}\n"
            f"⚡ Workload Preference: {c.workload_preference.value}\n"
            f"📆 Preferred Days: {', '.join(c.preferred_days)}\n"
            f"📚 Focus Courses: {', '.join(c.course_codes) if c.course_codes else 'All courses'}\n\n"
            f"Shall I generate your study plan now? (yes/no)"
        )

    def generate_study_plan(self,
                            snippets: Optional[List[Dict[str, Any]]] = None,
                            ollama_client=None,
                            model: str = "gemma3:4b") -> Dict[str, Any]:
        """
        Generate study plan based on collected constraints.

        Args:
            snippets: Course information snippets (optional, for retrieval)
            ollama_client: Ollama client for generation
            model: Ollama model name

        Returns:
            Dict with study plan and metadata
        """
        if self.user_constraints is None:
            return {
                "status": "error",
                "message": "No constraints collected. Call start_study_plan_flow() first.",
            }

        # Retrieve relevant course information if snippets provided
        course_context = ""
        if snippets and self.retrieval_service:
            # Search for relevant course information
            query = f"course workload difficulty schedule {' '.join(self.user_constraints.course_codes)}"
            retrieved = self.retrieval_service(query, snippets, top_k=5)

            course_context = "\n\n".join([
                f"Source: {snippets[idx].get('file_name', 'Unknown')}\n{snippets[idx].get('text', '')[:500]}"
                for _, _, idx in retrieved
            ])

        # Build specialized prompt
        prompt = self._build_study_plan_prompt(course_context)

        # Generate plan with Ollama
        study_plan_text = ""
        if ollama_client:
            try:
                response = ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7, "num_predict": 1024},
                )
                study_plan_text = response.get(
                    "message", {}).get("content", "")
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to generate study plan: {str(e)}",
                }
        else:
            # Direct API call
            import requests
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 1024},
                    },
                    timeout=120,
                )
                response.raise_for_status()
                study_plan_text = response.json().get("message", {}).get("content", "")
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to generate study plan: {str(e)}",
                }

        self.conversation_state = "complete"

        return {
            "status": "success",
            "study_plan": study_plan_text,
            "constraints": self.user_constraints.to_dict(),
            "prompt_used": prompt,
        }

    def _build_study_plan_prompt(self, course_context: str = "") -> str:
        """Build specialized prompt for study plan generation."""
        c = self.user_constraints

        # Adjust hours based on workload preference
        hours = c.available_hours_per_week or 20
        if c.workload_preference == WorkloadPreference.LIGHT:
            hours = int(hours * 0.7)
        elif c.workload_preference == WorkloadPreference.HEAVY:
            hours = int(hours * 1.3)

        prompt = f"""You are an expert academic advisor at HKBU. Create a personalized weekly study plan based on the following student constraints:

STUDENT CONSTRAINTS:
- Available Time: {hours} hours per week
- Goals: {', '.join(c.goals) if c.goals else 'General academic success'}
- Workload Preference: {c.workload_preference.value}
- Preferred Study Days: {', '.join(c.preferred_days)}
- Focus Courses: {', '.join(c.course_codes) if c.course_codes else 'All enrolled courses'}
"""

        if course_context:
            prompt += f"\n\nRELEVANT COURSE INFORMATION:\n{course_context}\n"

        prompt += """\n
STUDY PLAN REQUIREMENTS:
1. Create a detailed weekly schedule with specific time allocations for each day
2. Balance workload across preferred days
3. Include time for: lectures, self-study, assignments, and revision
4. Add specific recommendations for achieving the stated goals
5. Consider the workload preference (light/moderate/heavy)

FORMAT:
Provide the plan in this structure:

## Weekly Study Schedule
[Day-by-day breakdown with time slots]

## Course-Specific Recommendations
[Advice for each focus course]

## Study Tips
[Personalized recommendations based on goals]

## Weekly Hour Breakdown
[Summary of hours per activity type]

Based on the above, here is your personalized study plan:
"""

        return prompt

    def is_study_plan_query(self, text: str) -> bool:
        """Check if user input is related to study plan generation."""
        study_keywords = [
            "study plan", "schedule", "weekly plan", "timetable",
            "how should i study", "plan my studies", "study schedule",
            "help me organize", "study routine", "academic plan",
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in study_keywords)

    def reset(self):
        """Reset the study plan manager state."""
        self.user_constraints = None
        self.conversation_state = "collecting_constraints"
