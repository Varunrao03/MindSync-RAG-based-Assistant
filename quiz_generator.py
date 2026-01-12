"""
Quiz Generator for RAG System
Generates quizzes from documents in the vector store using the RAG system
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_openai import ChatOpenAI


class QuizGenerator:
    """Generates quizzes from documents in the RAG vector store."""
    
    def __init__(self):
        """Initialize the quiz generator with LLM."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        # Strip whitespace in case of formatting issues
        openai_api_key = openai_api_key.strip()
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        self.output_dir = Path(__file__).parent / "Generated_Quizzes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_quiz_from_context(
        self,
        context: str,
        num_questions: int = 5,
        question_types: List[str] = None,
        difficulty: str = "medium",
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a quiz from provided context text.
        
        Args:
            context: The text content to generate quiz questions from
            num_questions: Number of questions to generate (default: 5)
            question_types: List of question types to include. 
                           Options: ['multiple_choice', 'true_false', 'short_answer']
                           If None, uses all types
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            topic: Optional topic/focus area for the quiz
        
        Returns:
            Dictionary containing the generated quiz
        """
        if question_types is None:
            question_types = ['multiple_choice', 'true_false', 'short_answer']
        
        # Create prompt for quiz generation
        question_types_str = ", ".join(question_types)
        topic_section = f"\nFocus Topic: {topic}\n" if topic else ""
        
        prompt = f"""You are an expert educator creating a {difficulty} level quiz.

Based on the following context, generate exactly {num_questions} quiz questions.

Question Types to Include: {question_types_str}
Difficulty Level: {difficulty}
{topic_section}
Context:
{context}

Generate a quiz with the following structure:
- For multiple choice questions: Provide the question, 4 options (A, B, C, D), and the correct answer (A, B, C, or D)
- For true/false questions: Provide the statement and indicate True or False as the correct answer
- For short answer questions: Provide the question and a brief model answer (2-3 sentences)

Format your response as a JSON object with the following structure:
{{
    "quiz_title": "Quiz Title",
    "difficulty": "{difficulty}",
    "topic": "{topic or 'General'}",
    "num_questions": {num_questions},
    "questions": [
        {{
            "question_number": 1,
            "question_type": "multiple_choice|true_false|short_answer",
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"] (only for multiple_choice),
            "correct_answer": "A" or "True" or "False" or "Answer text" (depending on type),
            "explanation": "Brief explanation of the answer"
        }},
        ...
    ]
}}

Only return the JSON object, no additional text."""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            if "```json" in response_text:
                # Extract JSON from markdown code block
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            # Parse JSON
            quiz_data = json.loads(response_text)
            
            # Add metadata
            quiz_data["generated_at"] = datetime.now().isoformat()
            quiz_data["source"] = "RAG System Quiz Generator"
            
            return quiz_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}...")
            raise ValueError(f"Failed to parse quiz JSON: {e}")
        except Exception as e:
            print(f"❌ Error generating quiz: {e}")
            raise
    
    def generate_quiz_from_query(
        self,
        rag_system,
        query: str,
        num_questions: int = 5,
        top_k: int = 3,
        question_types: List[str] = None,
        difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """Generate a quiz based on a query using the RAG system.
        
        Args:
            rag_system: The RAGSystem instance
            query: Query string to retrieve relevant context
            num_questions: Number of questions to generate
            top_k: Number of chunks to retrieve for context
            question_types: List of question types (default: all types)
            difficulty: Difficulty level
        
        Returns:
            Dictionary containing the generated quiz
        """
        # Retrieve relevant context using RAG
        results = rag_system.retriever.retrieve(query, top_k=top_k, score_threshold=0.0)
        
        if not results:
            raise ValueError("No relevant documents found for the query.")
        
        # Format context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(results, 1):
            doc_name = chunk.get('metadata', {}).get('document_name', 'Unknown')
            context_parts.append(f"--- Excerpt {i} from {doc_name} ---")
            context_parts.append(chunk['document'])
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Generate quiz from context
        quiz = self.generate_quiz_from_context(
            context=context,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            topic=query
        )
        
        return quiz
    
    def save_quiz(self, quiz: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save quiz to a JSON file.
        
        Args:
            quiz: The quiz dictionary to save
            filename: Optional custom filename. If None, generates one
        
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' 
                                for c in quiz.get('quiz_title', 'quiz'))[:50]
            safe_title = safe_title.replace(' ', '_')
            filename = f"quiz_{safe_title}_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(quiz, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Quiz saved to: {file_path}")
        return file_path
    
    def save_quiz_as_text(self, quiz: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save quiz as a formatted text file.
        
        Args:
            quiz: The quiz dictionary to save
            filename: Optional custom filename. If None, generates one
        
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' 
                                for c in quiz.get('quiz_title', 'quiz'))[:50]
            safe_title = safe_title.replace(' ', '_')
            filename = f"quiz_{safe_title}_{timestamp}.txt"
        
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write(f"{quiz.get('quiz_title', 'Quiz').upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Difficulty: {quiz.get('difficulty', 'N/A').upper()}\n")
            f.write(f"Topic: {quiz.get('topic', 'N/A')}\n")
            f.write(f"Number of Questions: {quiz.get('num_questions', 0)}\n")
            f.write(f"Generated: {quiz.get('generated_at', 'N/A')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Write questions
            questions = quiz.get('questions', [])
            for q in questions:
                q_num = q.get('question_number', 0)
                q_type = q.get('question_type', 'unknown')
                question_text = q.get('question', '')
                
                f.write(f"Question {q_num} ({q_type.replace('_', ' ').title()})\n")
                f.write("-" * 60 + "\n")
                f.write(f"{question_text}\n\n")
                
                # Write options for multiple choice
                if q_type == 'multiple_choice' and 'options' in q:
                    options = q['options']
                    for i, option in enumerate(options):
                        f.write(f"  {chr(65 + i)}. {option}\n")
                    f.write("\n")
                
                # Write correct answer
                correct = q.get('correct_answer', 'N/A')
                f.write(f"Correct Answer: {correct}\n")
                
                # Write explanation if available
                if 'explanation' in q:
                    f.write(f"Explanation: {q['explanation']}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"✅ Quiz saved as text to: {file_path}")
        return file_path
    
    def generate_and_save_quiz(
        self,
        rag_system,
        query: str,
        num_questions: int = 5,
        save_format: str = "both",
        **kwargs
    ) -> Dict[str, Path]:
        """Generate quiz and save it in the specified format(s).
        
        Args:
            rag_system: The RAGSystem instance
            query: Query string to retrieve relevant context
            num_questions: Number of questions to generate
            save_format: 'json', 'text', or 'both'
            **kwargs: Additional arguments for generate_quiz_from_query
        
        Returns:
            Dictionary with paths to saved files (keys: 'json', 'text')
        """
        # Generate quiz
        quiz = self.generate_quiz_from_query(
            rag_system=rag_system,
            query=query,
            num_questions=num_questions,
            **kwargs
        )
        
        saved_paths = {}
        
        # Save in requested format(s)
        if save_format in ('json', 'both'):
            json_path = self.save_quiz(quiz)
            saved_paths['json'] = json_path
        
        if save_format in ('text', 'both'):
            text_path = self.save_quiz_as_text(quiz)
            saved_paths['text'] = text_path
        
        return saved_paths


def main():
    """Example usage of the Quiz Generator."""
    from rag_system import RAGSystem
    
    print("=" * 60)
    print("Quiz Generator - Example Usage")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        print("\n📦 Initializing RAG System...")
        rag = RAGSystem()
        
        # Initialize quiz generator
        print("\n📝 Initializing Quiz Generator...")
        quiz_gen = QuizGenerator()
        
        # Example: Generate quiz from a query
        query = input("\nEnter a topic for quiz generation (or press Enter for default): ").strip()
        if not query:
            query = "machine learning fundamentals"
        
        num_questions = input("Number of questions (default 5): ").strip()
        num_questions = int(num_questions) if num_questions.isdigit() else 5
        
        print(f"\n🔄 Generating quiz on '{query}'...")
        saved_paths = quiz_gen.generate_and_save_quiz(
            rag_system=rag,
            query=query,
            num_questions=num_questions,
            save_format='both',
            difficulty='medium'
        )
        
        print("\n✅ Quiz generation complete!")
        print(f"   JSON: {saved_paths.get('json', 'N/A')}")
        print(f"   Text: {saved_paths.get('text', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

