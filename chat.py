from predictor import predict_category
from intent import detect_intent
from graph_generator import generate_count_plot
from graph_generator import generate_pie_chart
from graph_generator import generate_histogram
from graph_intent import detect_graph


def chat(prompt, model, vectorizer):

    lower_prompt = prompt.lower().strip()
    intent = detect_intent(lower_prompt)
    graphs = detect_graph(lower_prompt)
    graph_files = []
    
    # greetings = [
    #     "hi",
    #     "hello",
    #     "hey",
    #     "good morning",
    #     "good evening"
    # ]

    # if lower_prompt in greetings:

    #     return {

    #         "reply":
    #         "Hello 👋 I am your AI Product Assistant. Ask me anything about products."

    #     }

    # if "help" in lower_prompt:

    #     return {

    #         "reply":
    #         """I can help you with
    #         • Product category prediction
    #         • Product search
    #         • Product recommendation
    #         • Product comparison
    #         • Product related questions"""
    #     }
    
    # # --------------------------------
    # # THANK YOU
    # # --------------------------------

    # if "thank" in lower_prompt or "thanks" in lower_prompt:

    #     return {
    #         "reply": "You're welcome 😊"
    #     }
    
    # # --------------------------------
    # # GOODBYE
    # # --------------------------------

    # if lower_prompt in ["bye", "goodbye", "see you"]:

    #     return {
    #         "reply": "Goodbye 👋 Have a great day!"
    #     }


    # -------------------
    # Greeting
    # -------------------

    if intent == "greeting":

        return {
            "reply":
            "Hello 👋 Welcome to the AI Product Intelligence Assistant.\n\n"
            "I can help you with:\n"
            "• Product category prediction\n"
            "• Product search\n"
            "• Product recommendations\n\n"
            "How can I assist you today?"
        }

    if intent == "help":
        return {
            "reply":
            "I can classify a product and explain the result. You can also request "
            "a category comparison or a product chart.\n\n"
            "For example: wireless noise-cancelling headphones or show a pie chart."
        }

    # -------------------
    # Thanks
    # -------------------

    if intent == "thanks":

        return {
            "reply":
            "You're welcome 😊 Happy to help!"
        }

    # -------------------
    # Goodbye
    # -------------------

    if intent == "goodbye":

        return {
            "reply":
            "Goodbye 👋 Have a wonderful day!"
        }

    for graph in graphs:
   
       # if graph=="count":
        if "count" in prompt or "countplot" in prompt or "count plot" in prompt:

            graph_files.append(generate_count_plot())

        #elif graph=="pie":
        if "pie" in prompt or "pie chart" in prompt:    

            graph_files.append(generate_pie_chart())

        if graph=="histogram":

            graph_files.append(generate_histogram())

        # elif graph == "scatter":
        #     graph_files.append(generate_scatter_plot())

        # elif graph == "bar":
        #     graph_files.append(generate_bar_chart())

        # elif graph == "line":
        #     graph_files.append(generate_line_chart())

        # elif graph == "box":
        #     graph_files.append(generate_box_plot())

        # elif graph == "violin":
        #     graph_files.append(generate_violin_plot())

        # elif graph == "heatmap":
        #     graph_files.append(generate_heatmap())

        # elif graph == "pairplot":
        #     graph_files.append(generate_pairplot())

        return {

            "reply":"Graph generated successfully.",

            "graphs":graph_files
        }

    prediction = predict_category(prompt, model, vectorizer)

    if prediction["needs_clarification"]:
        alternatives = ", ".join(
            item["category"] for item in prediction["top_categories"]
        )
        return {
            "reply":
            "I'm not fully sure which category fits this product. Could you add "
            "more detail, such as its type or main use? Possible categories: "
            f"{alternatives}.",
            "predicted_category": prediction["predicted_category"],
            "confidence": prediction["confidence"],
            "top_categories": prediction["top_categories"],
            "needs_clarification": True
        }

    return {

        "reply":

        f"Based on your prompt, this belongs to '{prediction['predicted_category']}' category.",

        "predicted_category":

        prediction["predicted_category"],

        "confidence":

        prediction["confidence"],

        "top_categories": prediction["top_categories"],

        "needs_clarification": False
    }