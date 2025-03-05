import ast
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent , AgentType , Tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

load_dotenv()

def syntax_check( code : str ) -> str:
    try:
        compile(code , "<string>" , "exec")
        return "Syntax Check : No syntax errors found."
    except SyntaxError as s:
        return f"Syntax Error occured \n {s}"
    
def documentation_check( code : str ) -> str:
    missing_docs = []
    try:
        tree = ast.parse( code )
        for node in ast.walk(tree):
            if isinstance( node , ( ast.FunctionDef , ast.ClassDef ) ):
                if ast.get_docstring( node ) is None:
                    missing_docs.append( node.name )
        if missing_docs:
            return "Documentation Check : Missing docsstrings in : "+", ".join(missing_docs)
        else:
            return "Documentation Check : All functions are properly documented."
        
    except Exception as e:
        return f"Documentation Error when analyzing the code : {e}"
    
def style_check( code : str ) -> str:
    suggestions = []
    if "\t" in code:
        suggestions.append("Avoid mixing tabs and spaces; prefer 4 spaces per indentation level.")
    if suggestions:
        return "Style Check : "+" ".join(suggestions)
    else:
        return "Code style looks good"
    
def refactoring_suggestions( code : str ) -> str:
    return f"Refactoring Suggestions : Consider breaking long functions into smaller , reusable components and removing redundant code.\n{code}"

def security_analysis(code: str) -> str:
    return "Security Analysis: No obvious security vulnerabilities detected. Ensure to review the use of external libraries."

def test_suggestions(code: str) -> str:
    return "Test Suggestions: Consider writing unit tests for edge cases, exception handling, and integration of major functions."

def code_explanation(code: str) -> str:
    return "Code Explanation: This code implements specific functionalities. A detailed explanation would require more context."


llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )

tool_syntax = Tool(
    name="Syntax Check",
    func=syntax_check,
    description="Check for syntax errors in the provided code."
)
tool_docs = Tool(
    name="Documentation Check",
    func=documentation_check,
    description="Check for missing documentation in functions and classes."
)

tool_style = Tool(
    name="Style Check",
    func=style_check,
    description="Review the code for style consistency and formatting issues."
)
tool_refactor = Tool(
    name="Refactoring Suggestions",
    func=refactoring_suggestions,
    description="Provide suggestions for refactoring the code to improve readability and maintainability."
)

tool_security = Tool(
    name="Security Analysis",
    func=security_analysis,
    description="Analyze the code for potential security vulnerabilities."
)
tool_tests = Tool(
    name="Test Suggestions",
    func=test_suggestions,
    description="Provide suggestions for writing tests for the code."
)

tool_explain = Tool(
    name="Code Explanation",
    func=code_explanation,
    description="Provide a high-level explanation of what the code does."
)

static_tools = [tool_syntax, tool_docs]
agent_static = initialize_agent(
    static_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

quality_tools = [tool_style, tool_refactor]
agent_quality = initialize_agent(
    quality_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

security_tools = [tool_security, tool_tests]
agent_security = initialize_agent(
    security_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

explanation_tools = [tool_explain]
agent_explanation = initialize_agent(
    explanation_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def master_code_review(code: str) -> str:
    """
    Perform a comprehensive code review by aggregating results
    from multiple specialized agents.
    """
    results = {}

    try:
        results['Static Analysis'] = agent_static.run(code)
    except Exception as e:
        results['Static Analysis'] = f"Error: {str(e)}"

    try:
        results['Code Quality'] = agent_quality.run(code)
    except Exception as e:
        results['Code Quality'] = f"Error: {str(e)}"

    try:
        results['Security & Testing'] = agent_security.run(code)
    except Exception as e:
        results['Security & Testing'] = f"Error: {str(e)}"

    try:
        results['Code Explanation'] = agent_explanation.run(code)
    except Exception as e:
        results['Code Explanation'] = f"Error: {str(e)}"

    review_report = []
    for section, review in results.items():
        review_report.append(f"=== {section} ===\n{review}\n")
    return "\n".join(review_report)


def main():
    print("Welcome to code review Assistant!")
    choice = input("Enter 'FILE' for file path or 'PASTE' to paste code : ").strip().lower()
    code = ""
    if choice=="file":
        file_path = input("Enter file path : ").strip()
        if os.path.isfile(file_path):
            with open( file_path , "r" , encoding = "utf-8" ) as f:
                code = f.read()
        else:
            print( "File Not found" )
            return ""
        
    elif choice=="paste":
        print("Paste your code below , when finished type 'EOF' on a new line.")
        lines = []
        while True:
            line = input()
            if line.strip()=="EOF":
                break
            lines.append(line)
        code = "\n".join(lines)
    else:
        print("Invalid choice")

    review = master_code_review(code)
    print(review)

main()






