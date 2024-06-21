from search_agent import vlm_rag_agent

def chat_with_agent():

    print("\n\t\t\t Loading Agent .... !\n")
    AGENT = vlm_rag_agent()

    prompt = ("\n\n\t|=>> Press the letter (I) to enter image_url, (C) to chat with the model, and (Q) to exit: \n\t")
    while True:
        user_input = input(prompt)

        if user_input.lower().startswith("q"):
            print("\n\t|=>> Received quit signal => Exiting...")
            return

        elif user_input.lower().startswith("i"):
            image_url = input("\n\t|=>> Enter Image URL: ")
            print("\t|=>> Image URL updated successfully. Press (C) to chat with the model")

        elif user_input.lower().startswith("c"):
            print("\n\tEntering Chat Session - CTRL-C to start afresh!")
            try:
                while True:
                    question = input("\n\t|=>> Enter your query: ")
                    AGENT.chat_with_agent(image_url,question)
            except KeyboardInterrupt:
                print("\n\t====================================================\n")
                continue
        else:
            print("\n\tInvalid key!")


if __name__ == "__main__":
    chat_with_agent()