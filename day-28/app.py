from typing import Dict , List , Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader , TextLoader , Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
load_dotenv()
console = Console()

def load_document(file_path:str)->List[str]:
    console.print(f"[bold blue]Loading document: {file_path}[/bold blue]")
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path=file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path=file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path=file_path)
    else:
        console.print(f"[bold red]Unsupported File Format : {file_path}[/bold red]")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100 )
    chunks = text_splitter.split_documents(documents)
    console.print(f"[green]Document loaded and split into {len(chunks)} chunks [/green]")
    texts = [chunk.page_content for chunk in chunks]
    return texts

def identify_contract_basics(text_chunks:List[str] , llm)-> Dict[str,Any]:
    console.print("[bold blue]Identifying contract basics...[/bold blue]")
    full_text = "\n\n".join(text_chunks[:10])
    basics_prompt = PromptTemplate(
        input_variables=["contract_text"],
        template="""
        You are a legal expert. Analyze the beginning of this legal contract and identify:

        CONTRACT TEXT:
        {contract_text}

        Please provide:
        1. Title of the contract
        2. Date of the contract (if mentioned)
        3. Type of contract (e.g., employment, service agreement, NDA)
        4. Parties involved (names and roles)
        5. Brief description of what this contract is about (1-2 sentences)
        """
    )
    basic_chain = LLMChain( llm=llm , prompt=basics_prompt , verbose=False )
    result = basic_chain.run(contract_text=full_text)
    return {
        "basics_text":result,
        "processed_at":datetime.now().strftime("%Y-%m-%d")
    }

def extract_key_terms( text_chunks:List[str] , llm )-> List[Dict[str,str]]:
    console.print("[bold blue]Extracting key terms....[/bold blue]")
    full_text = "\n\n".join(text_chunks)
    terms_prompt = PromptTemplate(
        input_variables=["contract_text"],
        template="""
        You are a legal expert. Extract the 10 most important defined terms from this contract.
        
        A defined term is usually indicated by quotes, capitalization, or explicit definitions.

        CONTRACT TEXT:
        {contract_text}

        For each defined term, provide:
        1. The term itself
        2. Its definition as given in the contract

        Format as:
        Term: [term]
        Definition: [definition]

        -----
        """
    )
    terms_chain = LLMChain( llm=llm , prompt=terms_prompt , verbose=False )
    result = terms_chain.run(contract_text=full_text)

    terms = []
    for block in result.split("-----"):
        block = block.strip()
        if not block:
            continue
        term_dict = {}
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("Term:"):
                term_dict["term"] = line[5:].strip()
            elif line.startswith("Definition:"):
                term_dict["definition"] = line[11:].strip()

        if "term" in term_dict and "definition" in term_dict:
            terms.append(term_dict)
        
    return terms

def identify_obligations(text_chunks:List[str] , llm) -> List[Dict[str,str]]:
    console.print("[bold blue]Identifying obligations.....[/bold blue]")

    full_text = "\n\n".join(text_chunks)
    obligations_prompt = PromptTemplate(
        input_variables=["contract_text"],
        template="""
        You are a legal expert. Identify the 8 most important obligations in this contract.
        
        An obligation is a requirement that one party must fulfill (words like "shall", "must", "agrees to").

        CONTRACT TEXT:
        {contract_text}

        For each obligation, provide:
        1. Which party is responsible
        2. What they must do
        3. Any deadline or timeframe (if specified)
        4. Risk level (Low, Medium, High) based on the potential consequences

        Format as:
        Party: [party responsible]
        Obligation: [description]
        Deadline: [deadline or N/A]
        Risk: [Low/Medium/High]

        -----
        """
    )
    obligations_chain = LLMChain(
        llm=llm , prompt=obligations_prompt , verbose=False
    )
    result= obligations_chain.run(contract_text=full_text)
    obligations = []

    for obligation in result.split("-----"):
        obligation = obligation.strip()
        if not obligation:
            continue
        obligation_dict = {}
        for line in obligation.split("\n"):
            line = line.strip()
            if line.startswith("Party:"):
                obligation_dict["party"] = line[6:].strip()
            elif line.startswith("Obligation:"):
                obligation_dict["description"] = line[11:].strip()
            elif line.startswith("Deadline:"):
                obligation_dict["deadline"] = line[9:].strip()
            elif line.startswith("Risk:"):
                obligation_dict["risk_level"] = line[5:].strip()
        if "party" in obligation_dict and "description" in obligation_dict:
            obligations.append(obligation_dict)
    
    return obligations

def identify_risks(text_chunks: List[str], llm) -> List[Dict[str, str]]:
    console.print("[bold blue]Identifying risks...[/bold blue]")
    
    full_text = "\n\n".join(text_chunks)
    
    risks_prompt = PromptTemplate(
        input_variables=["contract_text"],
        template="""
        You are a legal expert. Identify the 6 most significant risks in this contract.
        
        A risk is a potential negative outcome or liability that one party might face.
        Look for clauses related to liability, indemnification, termination, penalties, etc.

        CONTRACT TEXT:
        {contract_text}

        For each risk, provide:
        1. Description of the risk
        2. Which party is most affected
        3. Severity (Low, Medium, High)
        4. A suggestion for how to mitigate this risk

        Format as:
        Description: [description]
        Affected Party: [party]
        Severity: [Low/Medium/High]
        Mitigation: [suggestion]

        ---
        """
    )
    
    risks_chain = LLMChain(
        llm=llm,
        prompt=risks_prompt,
        verbose=False
    )
    
    result = risks_chain.run(contract_text=full_text)
    
    risks = []
    for risk_block in result.split('---'):
        risk_block = risk_block.strip()
        if not risk_block:
            continue
        
        risk_dict = {}
        for line in risk_block.split('\n'):
            line = line.strip()
            if line.startswith("Description:"):
                risk_dict["description"] = line[12:].strip()
            elif line.startswith("Affected Party:"):
                risk_dict["affected_party"] = line[15:].strip()
            elif line.startswith("Severity:"):
                risk_dict["severity"] = line[9:].strip()
            elif line.startswith("Mitigation:"):
                risk_dict["mitigation"] = line[11:].strip()
        
        if "description" in risk_dict and "affected_party" in risk_dict:
            risks.append(risk_dict)
    
    return risks


def simplify_contract(text_chunks: List[str], llm) -> Dict[str, Any]:
    console.print("[bold blue]Simplifying Contract.....[/bold blue]")
    full_text = "\n\n".join(text_chunks)
    
    prompt = PromptTemplate(
        input_variables=["contract_text"],
        template="""
        You are a legal expert who simplifies complex legal contracts for non-lawyers.
        
        Simplify the following contract:
        
        CONTRACT:
        {contract_text}
        
        Provide:
        1. A simple explanation (2-3 paragraphs)
        2. Five key points
        3. Definitions of complex legal terms
        
        Format:
        
        # SIMPLE EXPLANATION
        [Your explanation]
        
        # KEY POINTS
        1. [First point]
        2. [Second point]
        ...
        
        # LEGAL TERMS EXPLAINED
        - [Term]: [Definition]
        - [Term]: [Definition]
        """
    )

    result = LLMChain(llm=llm, prompt=prompt, verbose=False).run(contract_text=full_text)

    sections = result.split("\n\n")

    explanation, key_points, terms_explained = "", [], {}

    section_type = None
    for line in sections:
        line = line.strip()

        if line.startswith("# SIMPLE EXPLANATION"):
            section_type = "explanation"
            continue
        elif line.startswith("# KEY POINTS"):
            section_type = "key_points"
            continue
        elif line.startswith("# LEGAL TERMS EXPLAINED"):
            section_type = "terms_explained"
            continue

        if section_type == "explanation":
            explanation += line + " "
        elif section_type == "key_points":
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                key_points.append(line.split(".", 1)[-1].strip())
        elif section_type == "terms_explained":
            if ":" in line:
                term, definition = line.split(":", 1)
                terms_explained[term.strip()] = definition.strip()

    return {
        "explanation": explanation.strip(),
        "key_points": key_points,
        "terms_explained": terms_explained
    }

def generate_recommendations(analysis_results: Dict[str, Any], llm) -> List[str]:
    console.print("[bold blue]Generating recommendations....[/bold blue]")
    basics = analysis_results.get("basics", {}).get("basics_text", "")
    risks_summary = "\n".join(
        f"- {risk.get('description', 'N/A')} (Severity: {risk.get('severity', 'N/A')})"
        for risk in analysis_results.get("risks", [])
    )
    
    obligations_summary = "\n".join(
        f"- {obligation.get('description', 'N/A')} (Party: {obligation.get('party', 'N/A')}, Risk: {obligation.get('risk_level', 'N/A')})"
        for obligation in analysis_results.get("obligations", [])
    )
    
    prompt = PromptTemplate(
        input_variables=["basics", "risks", "obligations"],
        template="""
        You are a legal expert. Based on this contract analysis, provide actionable recommendations.

        CONTRACT BASICS:
        {basics}
        
        KEY RISKS:
        {risks}
        
        KEY OBLIGATIONS:
        {obligations}
        
        Provide 5 specific, actionable recommendations to mitigate risks and ensure compliance.
        Format each recommendation starting with a dash (-).
        """
    )

    result = LLMChain(llm=llm, prompt=prompt, verbose=False).run(
        basics=basics,
        risks=risks_summary,
        obligations=obligations_summary
    )

    recommendations = []
    for line in result.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            recommendations.append(line[1:].strip())

    return recommendations

def display_analysis_results(analysis_results:Dict[str , Any]):
    console.print("[bold green]Contract Analysis Results[/bold green]")
    if "basics" in analysis_results:
        console.print(Panel(analysis_results["basics"]["basics_text"],title="[bold]Contract Basics[/bold]",border_style="blue"))
    if "simplified" in analysis_results:
        simplified = analysis_results["simplified"]
        console.print(Panel(simplified["explanation"] , title="[bold]Simple Explanation[/bold] " , border_style="green"))
        for i,point in enumerate(simplified["key_points"] , 1):
            console.print(f" {i}. {point}")
        if simplified["terms_explained"]:
            console.print("[bold]Legal Terms Explained[/bold]")
            terms_table = Table(show_header=True, header_style="bold")
            terms_table.add_column("Term", style="cyan")
            terms_table.add_column("Simple Explanation")
            
            for term, explanation in simplified["terms_explained"].items():
                terms_table.add_row(term, explanation)
            
            console.print(terms_table)

    if "terms" in analysis_results and analysis_results["terms"]:
        console.print("\n[bold]Key Defined Terms:[/bold]")
        terms_table = Table(show_header=True, header_style="bold")
        terms_table.add_column("Term", style="cyan")
        terms_table.add_column("Definition")
        
        for term in analysis_results["terms"]:
            terms_table.add_row(term["term"], term["definition"])
        
        console.print(terms_table)
    
    if "obligations" in analysis_results and analysis_results["obligations"]:
        console.print("\n[bold]Key Obligations:[/bold]")
        obligations_table = Table(show_header=True, header_style="bold")
        obligations_table.add_column("Party", style="cyan")
        obligations_table.add_column("Obligation")
        obligations_table.add_column("Deadline")
        obligations_table.add_column("Risk Level", justify="center")
        
        for obligation in analysis_results["obligations"]:
            obligations_table.add_row(
                obligation.get("party", "N/A"),
                obligation.get("description", "N/A"),
                obligation.get("deadline", "N/A"),
                f"[{'red' if obligation.get('risk_level') == 'High' else 'yellow' if obligation.get('risk_level') == 'Medium' else 'green'}]{obligation.get('risk_level', 'N/A')}[/]"
            )
        
        console.print(obligations_table)
    
    if "risks" in analysis_results and analysis_results["risks"]:
        console.print("\n[bold]Key Risks:[/bold]")
        risks_table = Table(show_header=True, header_style="bold")
        risks_table.add_column("Risk Description")
        risks_table.add_column("Affected Party", style="cyan")
        risks_table.add_column("Severity", justify="center")
        risks_table.add_column("Mitigation")
        
        for risk in analysis_results["risks"]:
            risks_table.add_row(
                risk.get("description", "N/A"),
                risk.get("affected_party", "N/A"),
                f"[{'red' if risk.get('severity') == 'High' else 'yellow' if risk.get('severity') == 'Medium' else 'green'}]{risk.get('severity', 'N/A')}[/]",
                risk.get("mitigation", "N/A")
            )
        
        console.print(risks_table)
    
    if "recommendations" in analysis_results and analysis_results["recommendations"]:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, recommendation in enumerate(analysis_results["recommendations"], 1):
            console.print(f"  {i}. {recommendation}")  




llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash" )
console.print("[bold yellow]Enter the input file path[/bold yellow]")
path = input("")
text_chunks = load_document(path)
analysis_results = {}
analysis_results["basics"] = identify_contract_basics(text_chunks, llm)
analysis_results["terms"] = extract_key_terms(text_chunks,llm)
analysis_results["obligations"] = identify_obligations(text_chunks , llm)
analysis_results["risks"] = identify_risks(text_chunks , llm)
analysis_results["simplified"] = simplify_contract(text_chunks , llm)
analysis_results["recommendations"] = generate_recommendations(analysis_results , llm)
display_analysis_results(analysis_results)
console.print("[bold green]Analysis Completed Successfully[/bold green]")