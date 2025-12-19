def compare_papers(papers_data: list):
    """
    Takes a list of dictionaries containing extracted structure.
    """
    comparison_table = []
    
    for p in papers_data:
        comparison_table.append({
            "Title": p.get('title', 'Unknown'),
            "Methodology": p.get('methodology', 'N/A'),
            "Results": p.get('results', 'N/A')
        })
        
    return comparison_table