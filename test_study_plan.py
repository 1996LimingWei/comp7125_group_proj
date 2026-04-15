"""
Test script for Study Plan Feature
"""
from Module2_LexicalRetrieval import lexical_search
from src.study_plan.manager import StudyPlanManager
import json
import sys
sys.path.insert(0, '.')


def load_snippets(path="./output/snippets.json"):
    """Load snippets from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load snippets: {e}")
        return []


def main():
    print("=" * 60)
    print("Study Plan Feature Test")
    print("=" * 60)

    # Load snippets
    snippets = load_snippets()
    print(f"\nLoaded {len(snippets)} snippets")

    if not snippets:
        print("No snippets found. Run data ingestion first.")
        return

    # Initialize Study Plan Manager
    manager = StudyPlanManager(retrieval_service=lexical_search)

    # Test 1: Start study plan flow
    print("\n" + "-" * 60)
    print("Test 1: Starting Study Plan Flow")
    print("-" * 60)
    response = manager.start_study_plan_flow()
    print(f"\nAssistant:\n{response}")

    # Test 2: Collect constraints
    print("\n" + "-" * 60)
    print("Test 2: Collecting Constraints")
    print("-" * 60)

    test_inputs = [
        "I can study 15 hours per week",
        "My goal is to pass all courses with good grades",
        "I prefer moderate workload",
        "I can study Monday to Friday",
        "I'm taking COMP7125 and DAAI courses",
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        result = manager.collect_constraint(user_input)
        print(f"Status: {result['status']}")
        print(f"Assistant: {result['message'][:200]}...")

        if result['status'] == 'ready':
            break

    # Test 3: Generate study plan
    print("\n" + "-" * 60)
    print("Test 3: Generating Study Plan")
    print("-" * 60)

    if manager.conversation_state == "ready_to_generate":
        print("\nGenerating study plan... (this may take a moment)\n")
        result = manager.generate_study_plan(
            snippets=snippets,
            model="gemma3:4b"
        )

        if result['status'] == 'success':
            print("Study Plan Generated Successfully!")
            print("\n" + "=" * 60)
            print(result['study_plan'][:1000] + "...")
        else:
            print(f"Error: {result['message']}")
    else:
        print("Not ready to generate. Missing constraints.")

    # Test 4: Check is_study_plan_query
    print("\n" + "-" * 60)
    print("Test 4: Study Plan Query Detection")
    print("-" * 60)

    test_queries = [
        "I need a study plan",
        "Help me schedule my studies",
        "What is the tuition fee?",
        "Create a weekly plan for me",
        "How do I organize my time?",
    ]

    for query in test_queries:
        is_study = manager.is_study_plan_query(query)
        print(f"  '{query}' -> {'STUDY PLAN' if is_study else 'REGULAR QUERY'}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
