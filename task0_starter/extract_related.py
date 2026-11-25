import re

def extract_related_titles():
    # 1. Read the related work tex file to find cite keys
    with open('base_paper/sec/2_related.tex', 'r') as f:
        related_tex = f.read()

    cite_keys = []
    # Find all \cite{key1,key2}
    matches = re.findall(r'\\cite\{([^}]+)\}', related_tex)
    for m in matches:
        keys = m.split(',')
        cite_keys.extend([k.strip() for k in keys])
    
    cite_keys = list(set(cite_keys))
    
    # 2. Read the bbl file to map keys to titles
    with open('base_paper/main.bbl', 'r') as f:
        bbl_content = f.read()
    
    # The bbl file has entries like:
    # \bibitem{key}
    # Authors.
    # \newblock Title.
    # \newblock In \emph{Venue}, ...
    
    # We can split by \bibitem
    bibitems = bbl_content.split('\\bibitem')[1:]
    
    key_to_title = {}
    for item in bibitems:
        # get the key which is in {}
        key_match = re.search(r'^\{([^}]+)\}', item)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        
        # In IEEEtran bbl, the title is usually after \newblock
        # Let's just find the text between the first \newblock and the second \newblock
        newblocks = item.split('\\newblock')
        if len(newblocks) >= 3:
            # newblocks[0] is authors
            # newblocks[1] is title
            title = newblocks[1].strip()
            # clean up the title (remove \emph, line breaks, etc)
            title = title.replace('\n', ' ')
            title = re.sub(r'\\emph\{([^}]+)\}', r'\1', title)
            title = title.strip('., ')
            key_to_title[key] = title
        else:
            # Fallback for different bbl formats
            pass
            
    related_titles = []
    for k in cite_keys:
        if k in key_to_title:
            related_titles.append(key_to_title[k])
            
    print("Papers cited in Base Paper's Related Work:")
    for t in related_titles:
        print("-", t)
        
    # Write to a file for easy access
    with open('base_paper_related_titles.txt', 'w') as f:
        for t in related_titles:
            f.write(t + "\n")

if __name__ == '__main__':
    extract_related_titles()
