from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
from agentic_rag_chatbot.crew import AgenticRagChatbot

# Initialize your chatbot
chatbot = AgenticRagChatbot()

# Define test case inputs
in_scope_input = {
    'query': "how do i add a widget in the dashboard?"
}
out_scope_input = {
    'query': "What's the weather in New York today?"
}
ambiguous_query = {
    'query': "What does it do?"
}

# Get test case outputs
in_scope_output = chatbot.crew().kickoff(inputs=in_scope_input)
out_scope_output = chatbot.crew().kickoff(inputs=out_scope_input)
ambiguous_output = chatbot.crew().kickoff(inputs=ambiguous_query)

# Define test cases with expected outputs
test_cases = [
    # In-scope test cases
    LLMTestCase(
        input=in_scope_input["query"],
        actual_output=in_scope_output.raw,
        context=[
            '''1. Click on the Dashboard Tab, located in the top navigation menu, and then click on the IT subtab. 2. Click on the Manage Views button, located in the upper right-hand side. Once clicked, a menu will expand with options to select from. 3. Click on the Add New Widget option from the menu that expands. Once clicked, the Create New Widget pop-up box will appear. 4. Select the desired Widget from the drop-down menu from the wide range of widget options displayed. 5. Once selected, the name of the Widget will appear inside the Widget Type field. Depending on the Widget selected, the Create New Widget pop-up box will expand to oﬀer additional visual setting options to select from. 6. Choose the appropriate visualization settings in the Create New Widget pop-up box and Click "OK" 7. The new widget will appear on the Dashboards Main Page (or any sub-tab where the user desires to add the Widget) automatically.'''],
        expected_output="A step by step response on how to add a widget using Aparavi",
        retrieval_context=[in_scope_output.tasks_output[0].raw]
    ),

    # Out-of-scope test cases
    LLMTestCase(
        input=out_scope_input["query"],
        actual_output=out_scope_output.raw,
        context=["No relevant context as this is out of scope"],
        expected_output="This system is designed specifically for Aparavi-related information and cannot be used for general-purpose questions like weather forecasts.",
        retrieval_context=[out_scope_output.tasks_output[0].raw]
    ),

    # Edge case: Ambiguous query
    LLMTestCase(
        input=ambiguous_query['query'],
        actual_output=ambiguous_output.raw,
        context=['''Aparavi's dashboard screen is designed to provide users with a comprehensive view of their data metrics. The dashboards are customizable and allow the user to see various widgets showing data metrics related to the files scanned by the system. Dashboards are broken into subtabs, such as Information and System.'''],
        expected_output="A response answers generally what Aparavi does",
        retrieval_context=[ambiguous_output.tasks_output[0].raw]
    )
]

# Define metrics to evaluate
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.8),
    HallucinationMetric(threshold=0.2)
]

# Run evaluation
results = evaluate(test_cases, metrics)

# Print results
for i, result in enumerate(results):
    print(f"\nTest Case {i+1}:")
    print(f"Input: {test_cases[i].input}")
    print(f"Actual Output: {test_cases[i].actual_output}")
    print(f"Expected Output: {test_cases[i].expected_output}")
    print("Metrics:")
    for metric_name, passed in result.items():
        print(f"  {metric_name}: {'Passed' if passed else 'Failed'}")
