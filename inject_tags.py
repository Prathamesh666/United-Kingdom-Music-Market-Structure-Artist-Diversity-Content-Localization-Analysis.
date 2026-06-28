import os
import streamlit as st

def patch_streamlit_head():
    # Locates the master HTML template on the hosted Streamlit container server
    index_path = os.path.join(os.path.dirname(st.__file__), "static", "index.html")
    
    # 1. Paste your explicit GA4 Tag (REPLACE G-XXXXXXXXXX WITH YOUR MEASUREMENT ID)
    ga_tag = """
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-H7VP3CYEPB"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-H7VP3CYEPB');
    </script>
    """
    
    # 2. Paste your explicit Search Console Tag (REPLACE WITH YOUR ACTUAL CODE FROM GSC)
    gsc_tag = '<meta name="google-site-verification" content="8qhJewqcfQuP-HpMtrPOHyc72ENL1xOzBI_THkMVHKo" />'
    
    # Combine the tags together
    combined_tags = f"\n{ga_tag}\n{gsc_tag}\n"
    
    # Read the existing server template file
    with open(index_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        
    # Inject both tags cleanly into the head container if they aren't already written
    if "googletagmanager" not in html_content:
        patched_html = html_content.replace("<head>", f"<head>{combined_tags}")
        with open(index_path, "w", encoding="utf-8") as file:
            file.write(patched_html)
        print("Successfully patched Streamlit index.html with tracking tags!")

if __name__ == "__main__":
    patch_streamlit_head()