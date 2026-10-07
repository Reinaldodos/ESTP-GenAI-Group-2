"""Interface en ligne de commande du pipeline RAG."""

from pathlib import Path

from rag_pipeline import RAGConfig, RAGPipeline


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def main() -> None:
    pipeline = RAGPipeline(RAGConfig.from_yaml(CONFIG_PATH))
    print(f"Pipeline prêt ({pipeline.chunk_count} passages indexés).")

    while True:
        question = input("\nQuestion (ou 'exit') : ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        response = pipeline.ask(question)
        print("\n--- Passages retrouvés ---")
        for source in response.sources:
            print(f"\n{source['source']} (score={source['score']:.3f})")
            print(source["text"])
        print("\n--- Réponse ---")
        print(response.answer)


if __name__ == "__main__":
    main()
