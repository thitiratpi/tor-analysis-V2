"""
================================================================================
WISESIGHT TOR ANALYZER & BUDGET ESTIMATION
Streamlit Web Application
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from docx import Document
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================================================================================
# PAGE CONFIG
# ================================================================================

st.set_page_config(
    page_title="Wisesight TOR Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS
# ================================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .product-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #dc3545;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #c82333;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def extract_number_of_users(text):
    """ดึงจำนวน users จาก TOR"""
    text_lower = text.lower()
    patterns = [
        r'(\d+)\s*(?:users?|คน|ผู้ใช้)',
        r'(?:users?|คน|ผู้ใช้)[:\s]*(\d+)',
        r'จำนวน(?:ผู้ใช้|คน)[:\s]*(\d+)',
        r'(?:สามารถใช้งานได้|รองรับ)[:\s]*(\d+)\s*(?:users?|คน|ผู้ใช้)'
    ]
    found_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                num = int(match)
                if 1 <= num <= 1000:
                    found_numbers.append(num)
            except:
                pass
    if found_numbers:
        return max(set(found_numbers), key=found_numbers.count)
    return 10

def extract_data_backward_days(text):
    """ดึงระยะเวลาย้อนหลังจาก TOR"""
    text_lower = text.lower()
    
    month_patterns = [
        r'(\d+)\s*(?:เดือน|month(?:s)?)',
        r'(?:ย้อนหลัง|backward|history)[:\s]*(\d+)\s*(?:เดือน|month(?:s)?)',
        r'(?:ข้อมูล|data)(?:ย้อนหลัง)?[:\s]*(\d+)\s*(?:เดือน|month(?:s)?)'
    ]
    
    for pattern in month_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                months = int(match)
                if 1 <= months <= 120:
                    return months * 30
            except:
                pass
    
    year_patterns = [
        r'(\d+)\s*(?:ปี|year(?:s)?)',
        r'(?:ย้อนหลัง|backward|history)[:\s]*(\d+)\s*(?:ปี|year(?:s)?)'
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                years = int(match)
                if 1 <= years <= 10:
                    return years * 365
            except:
                pass
    
    day_patterns = [
        r'(\d+)\s*(?:วัน|day(?:s)?)',
        r'(?:ย้อนหลัง|backward|history)[:\s]*(\d+)\s*(?:วัน|day(?:s)?)'
    ]
    
    for pattern in day_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                days = int(match)
                if 1 <= days <= 3650:
                    return days
            except:
                pass
    
    return 90

def extract_social_channels(text):
    """ดึง social channels จาก TOR"""
    text_lower = text.lower()
    channels = {
        'facebook': 0, 'instagram': 0, 'twitter': 0, 'line': 0,
        'tiktok': 0, 'youtube': 0, 'pantip': 0, 'email': 0,
        'webchat': 0, 'live_chat': 0, 'application': 0
    }
    
    channel_patterns = {
        'facebook': [r'facebook', r'fb', r'เฟซบุ๊ก'],
        'instagram': [r'instagram', r'ig', r'อินสตาแกรม'],
        'twitter': [r'twitter', r'x\.com', r'ทวิตเตอร์'],
        'line': [r'\bline\b', r'ไลน์'],
        'tiktok': [r'tiktok', r'ติ๊กต๊อก'],
        'youtube': [r'youtube', r'ยูทูป'],
        'pantip': [r'pantip', r'พันทิป'],
        'webchat': [r'webchat', r'web\s*chat', r'แชทบนเว็บ'],
        'live_chat': [r'live\s*chat', r'แชทสด'],
        'application': [r'application', r'mobile\s*app', r'แอปพลิเคชัน', r'แอป'],
        'email': [r'email', r'e-mail', r'อีเมล']
    }
    
    for channel, patterns in channel_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                channels[channel] = 1
                break
    
    return channels

def count_warroom_connect_channels(channels):
    """นับ Warroom Connect channels"""
    connect_channels = ['webchat', 'live_chat', 'application']
    return sum(channels.get(ch, 0) for ch in connect_channels)

def clean_text(text):
    """ทำความสะอาด text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_sentences_from_tor(text, min_length=10):
    """แยกประโยคจาก TOR"""
    text = clean_text(text)
    paragraphs = text.split('\n')
    sentences = []
    
    for paragraph in paragraphs:
        sents = re.split(r'[.!?]+\s+', paragraph)
        sentences.extend(sents)
    
    filtered_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if (len(sent) >= min_length and not sent.isdigit() and 
            sent.lower() != 'nan' and len(sent.split()) >= 3):
            filtered_sentences.append(sent)
    
    seen = set()
    unique_sentences = []
    for sent in filtered_sentences:
        if sent not in seen:
            seen.add(sent)
            unique_sentences.append(sent)
    
    return unique_sentences

def read_file_content(uploaded_file):
    """อ่านเนื้อหาจากไฟล์ที่ upload"""
    content = ""
    try:
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(io.BytesIO(uploaded_file.read()))
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return content

@st.cache_resource
def load_semantic_model():
    """Load และ cache AI model"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calculate_semantic_similarity_with_id(tor_sentences, product_sentence, model):
    """คำนวณ similarity"""
    tor_embeddings = model.encode(tor_sentences)
    product_embedding = model.encode([product_sentence])
    similarities = cosine_similarity(product_embedding, tor_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    similarity_score = ((best_similarity + 1) / 2) * 100
    return similarity_score, best_match_idx

def format_currency(value):
    """Format ตัวเลขเป็นรูปแบบเงิน"""
    if pd.isna(value) or value == 0:
        return "0"
    return f"{int(value):,}"

# ================================================================================
# ANALYSIS FUNCTIONS
# ================================================================================

def analyze_tor(tor_text, requirements_df, threshold=60.0):
    """วิเคราะห์ TOR ด้วย Semantic Similarity"""
    
    # Validate DataFrame
    requirements_df = requirements_df.dropna(subset=['Product', 'Category'])
    requirements_df['Product'] = requirements_df['Product'].astype(str).str.strip()
    requirements_df = requirements_df[requirements_df['Product'] != 'nan']
    
    # Extract TOR sentences
    tor_sentences = extract_sentences_from_tor(tor_text, min_length=10)
    if len(tor_sentences) == 0:
        return None, 0, []
    
    # Load model
    model = load_semantic_model()
    
    # Analysis
    product_unique_matches = {}
    results = []
    
    for idx, row in requirements_df.iterrows():
        product = row['Product']
        category = row['Category']
        
        sentence_th = str(row.get('Sentence (TH)', '')) if 'Sentence (TH)' in requirements_df.columns else ""
        sentence_eng = str(row.get('Sentence (ENG)', '')) if 'Sentence (ENG)' in requirements_df.columns else ""
        
        product_sentence = sentence_eng if sentence_eng and sentence_eng != 'nan' else sentence_th
        if not product_sentence or product_sentence == 'nan':
            continue
        
        similarity_score, matched_tor_id = calculate_semantic_similarity_with_id(
            tor_sentences, product_sentence, model
        )
        
        matched = similarity_score >= threshold
        
        results.append({
            'Product': product,
            'Category': category,
            'Sentence (TH)': sentence_th,
            'Sentence (ENG)': sentence_eng,
            'Similarity (%)': round(similarity_score, 2),
            'Matched': matched,
            'Matched_TOR_ID': matched_tor_id
        })
        
        # Unique TOR Matches
        if product not in product_unique_matches:
            product_unique_matches[product] = {}
        
        if matched_tor_id not in product_unique_matches[product]:
            product_unique_matches[product][matched_tor_id] = {
                'similarity': similarity_score,
                'category': category,
                'matched': matched
            }
        else:
            if similarity_score > product_unique_matches[product][matched_tor_id]['similarity']:
                product_unique_matches[product][matched_tor_id] = {
                    'similarity': similarity_score,
                    'category': category,
                    'matched': matched
                }
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Similarity (%)', ascending=False).reset_index(drop=True)
    
    # Calculate scores
    product_scores = {}
    for product in results_df['Product'].unique():
        product_data = results_df[results_df['Product'] == product]
        unique_matches = product_unique_matches[product]
        
        unique_similarities = [m['similarity'] for m in unique_matches.values()]
        avg_similarity = np.mean(unique_similarities) if unique_similarities else 0
        
        matched_sentences = product_data[product_data['Matched'] == True]
        num_matched = len(matched_sentences)
        num_total = len(product_data)
        match_rate = (num_matched / num_total) * 100 if num_total > 0 else 0
        
        matched_categories = matched_sentences['Category'].nunique() if num_matched > 0 else 0
        total_categories = product_data['Category'].nunique()
        category_coverage = (matched_categories / total_categories) * 100 if total_categories > 0 else 0
        
        overall_score = (avg_similarity * 0.55) + (match_rate * 0.35) + (category_coverage * 0.10)
        
        product_scores[product] = {
            'match_rate': match_rate,
            'avg_similarity': avg_similarity,
            'category_coverage': category_coverage,
            'overall_score': overall_score,
            'num_matched': num_matched,
            'num_total': num_total,
            'unique_tor_matches': len(unique_matches)
        }
    
    # Select products
    matched_products = []
    MIN_AVG_SIMILARITY = 50.0
    MIN_MATCH_RATE = 30.0
    
    for product, scores in product_scores.items():
        if scores['avg_similarity'] >= MIN_AVG_SIMILARITY or scores['match_rate'] >= MIN_MATCH_RATE:
            matched_products.append(product)
    
    if len(matched_products) == 0:
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        if len(sorted_products) > 0:
            matched_products.append(sorted_products[0][0])
    
    return results_df, len(tor_sentences), matched_products, product_scores

def calculate_budget(requirements, matched_products, pricing_df, addon_df):
    """คำนวณงบประมาณ"""
    budgets = []
    
    for product in matched_products:
        if product == 'Zocial Eye':
            budget = calculate_zocial_eye_budget(requirements, pricing_df, addon_df)
            budgets.append(budget)
        elif product == 'Warroom':
            budget = calculate_warroom_budget(requirements, pricing_df, addon_df)
            budgets.append(budget)
    
    return budgets

def calculate_zocial_eye_budget(requirements, pricing_df, addon_df):
    """คำนวณงบประมาณ Zocial Eye"""
    num_users = requirements['num_users']
    data_backward = requirements['data_backward_days']
    
    ze_packages = pricing_df[pricing_df['Product'] == 'Zocial Eye'].copy()
    suitable_packages = ze_packages[
        (ze_packages['Data_Backward (Days)'] >= data_backward) &
        (ze_packages['User_Limit (User)'] >= num_users)
    ]
    
    if suitable_packages.empty:
        suitable_packages = ze_packages
    
    suitable_packages = suitable_packages.sort_values('Total_Price_Per_Year (THB)')
    selected_package = suitable_packages.iloc[0] if len(suitable_packages) > 0 else ze_packages.iloc[-1]
    
    package_details = selected_package.to_dict()
    
    return {
        'Product': 'Zocial Eye',
        'Package': selected_package['Package'],
        'Package_Details': package_details,
        'User_Limit': int(selected_package['User_Limit (User)']),
        'Required_Users': num_users,
        'Data_Backward_Limit': int(selected_package['Data_Backward (Days)']),
        'Required_Data_Backward': data_backward,
        'Base_Price': selected_package['Total_Price_Per_Year (THB)'],
        'Total_Price': selected_package['Total_Price_Per_Year (THB)'],
        'Meets_Requirements': num_users <= selected_package['User_Limit (User)'] and 
                             data_backward <= selected_package['Data_Backward (Days)']
    }

def calculate_warroom_budget(requirements, pricing_df, addon_df):
    """คำนวณงบประมาณ Warroom"""
    num_users = requirements['num_users']
    warroom_connect_channels = requirements['warroom_connect_channels']
    
    wr_packages = pricing_df[pricing_df['Product'] == 'Warroom'].copy()
    suitable_packages = wr_packages[wr_packages['User_Limit (User)'] >= num_users]
    
    if suitable_packages.empty:
        suitable_packages = wr_packages
    
    suitable_packages = suitable_packages.sort_values('Total_Price_Per_Year (THB)')
    selected_package = suitable_packages.iloc[0] if len(suitable_packages) > 0 else wr_packages.iloc[-1]
    
    package_details = selected_package.to_dict()
    base_price = selected_package['Total_Price_Per_Year (THB)']
    
    # Calculate Warroom Connect cost
    warroom_connect_cost = 0
    if warroom_connect_channels > 0:
        connect_addon = addon_df[
            (addon_df['Product'] == 'Warroom') & 
            (addon_df['AddOn_Name'] == 'Warroom Connect')
        ]
        if not connect_addon.empty:
            connect_price = connect_addon.iloc[0]['Price (THB)']
            warroom_connect_cost = warroom_connect_channels * connect_price
    
    total_price = base_price + warroom_connect_cost
    
    return {
        'Product': 'Warroom',
        'Package': selected_package['Package'],
        'Package_Details': package_details,
        'User_Limit': int(selected_package['User_Limit (User)']),
        'Required_Users': num_users,
        'Warroom_Connect_Channels': warroom_connect_channels,
        'Base_Price': base_price,
        'Warroom_Connect_Cost': warroom_connect_cost,
        'Total_Price': total_price,
        'Meets_Requirements': num_users <= selected_package['User_Limit (User)']
    }

# ================================================================================
# MAIN APP
# ================================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Wisesight TOR Analyzer & Budget</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Semantic Analysis & Budget Estimation</p>', unsafe_allow_html=True)
    
    # ================================================================================
    # SIDEBAR - FILE UPLOAD
    # ================================================================================
    
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # Upload TOR
        st.subheader("Upload TOR")
        tor_file = st.file_uploader(
            "Drag and drop file here",
            type=['txt', 'docx', 'pdf'],
            help="Limit 200MB per file • PDF, DOCX, TXT",
            key="tor_upload"
        )
        
        if tor_file:
            st.success(f"✓ {tor_file.name} ({tor_file.size/1024:.1f}KB)")
        
        st.markdown("---")
        
        # Upload Product Requirements
        st.subheader("Product Reqs (Excel)")
        req_file = st.file_uploader(
            "Drag and drop file here",
            type=['xlsx'],
            help="Limit 200MB per file • XLSX",
            key="req_upload"
        )
        
        if req_file:
            st.success(f"✓ {req_file.name} ({req_file.size/1024:.1f}KB)")
        
        st.markdown("---")
        
        # Upload Pricing
        st.subheader("Pricing (Excel)")
        pricing_file = st.file_uploader(
            "Drag and drop file here",
            type=['xlsx'],
            help="Limit 200MB per file • XLSX",
            key="pricing_upload"
        )
        
        if pricing_file:
            st.success(f"✓ {pricing_file.name} ({pricing_file.size/1024:.1f}KB)")
    
    # ================================================================================
    # MAIN CONTENT
    # ================================================================================
    
    # Check if all files uploaded
    all_files_uploaded = all([tor_file, req_file, pricing_file])
    
    if all_files_uploaded:
        st.markdown('<div class="success-box">✓ Files Loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ️ Please upload all required files in the sidebar</div>', unsafe_allow_html=True)
        return
    
    # Start Analysis Button
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Analyzing... This may take a few minutes."):
            
            try:
                # Read files
                tor_content = read_file_content(tor_file)
                requirements_df = pd.read_excel(req_file)
                pricing_df = pd.read_excel(pricing_file, sheet_name='Product_Pricing')
                
                # Try to load addon, if not found create empty
                try:
                    addon_df = pd.read_excel(pricing_file, sheet_name='AddOn_Feature')
                except:
                    addon_df = pd.DataFrame(columns=['Product', 'AddOn_Name', 'Price (THB)'])
                
                # Extract requirements
                requirements = {
                    'num_users': extract_number_of_users(tor_content),
                    'data_backward_days': extract_data_backward_days(tor_content),
                    'channels': extract_social_channels(tor_content),
                    'warroom_connect_channels': count_warroom_connect_channels(
                        extract_social_channels(tor_content)
                    )
                }
                
                # Analyze TOR
                results_df, total_sentences, matched_products, product_scores = analyze_tor(
                    tor_content, requirements_df, threshold=60.0
                )
                
                if results_df is None or len(matched_products) == 0:
                    st.error("❌ Unable to match TOR with any products")
                    return
                
                # Calculate budgets
                budgets = calculate_budget(requirements, matched_products, pricing_df, addon_df)
                
                # Store in session state
                st.session_state['analysis_complete'] = True
                st.session_state['results_df'] = results_df
                st.session_state['matched_products'] = matched_products
                st.session_state['product_scores'] = product_scores
                st.session_state['budgets'] = budgets
                st.session_state['requirements'] = requirements
                
                st.success("✅ Complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
                return
    
    # ================================================================================
    # DISPLAY RESULTS
    # ================================================================================
    
    if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2 = st.tabs(["🏆 Results & Budget", "📋 Details"])
        
        with tab1:
            matched_products = st.session_state['matched_products']
            budgets = st.session_state['budgets']
            requirements = st.session_state['requirements']
            product_scores = st.session_state['product_scores']
            
            # Summary box
            st.markdown(f"""
            <div class="success-box">
                <h3>Matched: {', '.join(matched_products)}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display each product
            for budget in budgets:
                product = budget['Product']
                scores = product_scores.get(product, {})
                package_details = budget['Package_Details']
                
                # Product Card
                st.markdown(f"""
                <div class="product-card">
                    <h2>{product}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Package",
                        budget['Package']
                    )
                
                with col2:
                    st.metric(
                        "Base Price",
                        f"{format_currency(budget['Base_Price'])} THB"
                    )
                
                with col3:
                    add_ons = budget.get('Warroom_Connect_Cost', 0)
                    st.metric(
                        "Add-ons Cost",
                        f"{format_currency(add_ons)} THB"
                    )
                
                with col4:
                    st.metric(
                        "Total Price",
                        f"{format_currency(budget['Total_Price'])} THB",
                        delta="per year"
                    )
                
                # Details in expander
                with st.expander(f"📋 Package Details - {budget['Package']}", expanded=True):
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**Package Information:**")
                        
                        # Display relevant fields
                        if product == 'Zocial Eye':
                            if 'User_Limit (User)' in package_details:
                                st.write(f"• User Limit: {int(package_details['User_Limit (User)'])} users")
                            if 'Data_Backward (Days)' in package_details:
                                days = int(package_details['Data_Backward (Days)'])
                                st.write(f"• Data Backward: {days} days ({days/30:.1f} months)")
                            if 'Campaign_Limit' in package_details and not pd.isna(package_details['Campaign_Limit']):
                                st.write(f"• Campaign Limit: {int(package_details['Campaign_Limit'])} campaigns")
                            if 'Message_Limit_Per_Contract (Messages)' in package_details:
                                st.write(f"• Message Limit: {format_currency(package_details['Message_Limit_Per_Contract (Messages)'])} messages")
                        
                        elif product == 'Warroom':
                            if 'User_Limit (User)' in package_details:
                                st.write(f"• User Limit: {int(package_details['User_Limit (User)'])} users")
                            if 'Message_Limit_Per_Month (Messages)' in package_details:
                                msg_limit = package_details['Message_Limit_Per_Month (Messages)']
                                if pd.isna(msg_limit) or str(msg_limit).lower() == 'unlimited':
                                    st.write(f"• Message Limit: unlimited")
                                else:
                                    st.write(f"• Message Limit: {format_currency(msg_limit)} messages/month")
                            if 'Chat_History' in package_details and not pd.isna(package_details['Chat_History']):
                                st.write(f"• Chat History: {package_details['Chat_History']}")
                    
                    with detail_col2:
                        st.markdown("**Requirements Status:**")
                        
                        # Users
                        if budget['Required_Users'] <= budget['User_Limit']:
                            st.success(f"✅ Users: {budget['Required_Users']} / {budget['User_Limit']} (satisfied)")
                        else:
                            st.warning(f"⚠️ Users: {budget['Required_Users']} / {budget['User_Limit']} (need upgrade)")
                        
                        # Data Backward (Zocial Eye only)
                        if product == 'Zocial Eye':
                            req_days = budget['Required_Data_Backward']
                            limit_days = budget['Data_Backward_Limit']
                            if req_days <= limit_days:
                                st.success(f"✅ Data Backward: {req_days} / {limit_days} days (satisfied)")
                            else:
                                st.warning(f"⚠️ Data Backward: {req_days} / {limit_days} days (need upgrade)")
                        
                        # Warroom Connect
                        if product == 'Warroom' and budget.get('Warroom_Connect_Channels', 0) > 0:
                            channels = requirements['channels']
                            connect_list = [ch for ch in ['webchat', 'live_chat', 'application'] 
                                          if channels.get(ch, 0) > 0]
                            st.info(f"🔗 Warroom Connect: {budget['Warroom_Connect_Channels']} channels ({', '.join(connect_list)})")
                
                st.markdown("---")
            
            # Summary if multiple products
            if len(budgets) > 1:
                st.markdown("""
                <div class="info-box">
                    <h3>💡 Recommendation</h3>
                    <p>Both products matched the TOR requirements. Each product serves different purposes:</p>
                    <ul>
                        <li><strong>Zocial Eye:</strong> Social listening & analytics</li>
                        <li><strong>Warroom:</strong> Customer engagement & chat management</li>
                    </ul>
                    <p>💡 Consider both products as they complement each other.</p>
                </div>
                """, unsafe_allow_html=True)
                
                total_investment = sum(b['Total_Price'] for b in budgets)
                st.metric("📊 Total Investment", f"{format_currency(total_investment)} THB/year")
            
            # Download Budget
            st.markdown("### 📥 Download Budget")
            
            for budget in budgets:
                product_name = budget['Product'].replace(' ', '_')
                budget_df = pd.DataFrame([budget])
                csv = budget_df.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label=f"⬇️ Download {budget['Product']} Budget",
                    data=csv,
                    file_name=f"budget_{product_name}.csv",
                    mime="text/csv"
                )
        
        with tab2:
            st.subheader("📋 Detailed Analysis Data")
            
            results_df = st.session_state['results_df']
            
            # Display dataframe
            st.dataframe(
                results_df,
                use_container_width=True,
                height=400
            )
            
            # Download
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="⬇️ Download Full Analysis (CSV)",
                data=csv,
                file_name="tor_analysis_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
