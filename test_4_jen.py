import streamlit as st
import openai
import os
from openai import OpenAI
import urllib.request
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time

# Configuration OpenAI
client = OpenAI(api_key=st.secrets["api_key"])

def generate_brand_messages(image_prompt):
    """Génère 3 propositions de messages de marque basés sur le prompt de l'image"""
    try:
        message_prompt = f"""You are a senior copywriter for MAIF insurance company, expert in their "assureur militant" campaign and their mission as "société à mission". 

Based on this specific image scene: "{image_prompt}"

Generate 3 short, impactful brand messages following MAIF's brand guidelines:

BRAND ESSENCE:
- Raison d'être: "Une attention sincère portée à l'autre et au monde pour un mieux commun"
- Assureur militant - engaged insurance company since 1934
- Société à mission with 5 objectives: sociétaires, collaborateurs, société solidaire, transition écologique, modèles d'entreprises engagées
- Modèle mutualiste - mutual model where each member is both insurer and insured
- Valeurs: humanisme, solidarité, démocratie
- Triangle rouge dynamique symbolisant l'engagement et l'ouverture

WRITING STYLE:
- Militant and engaged, but warm and human
- Clear and accessible language (never corporate)
- Sincere attention and active listening
- Personal pronouns (you, your, we, together)
- Focus on commitment and responsibility
- Maximum 5-6 words per message (very short and punchy)
- Maximum 2 lines per message
- Reflect mutualiste spirit and solidarity

CONTEXT ADAPTATION:
- If family scene → focus on mutual protection, solidarity, engagement
- If home scene → focus on secure community, mutual responsibility
- If life moments → focus on sincere attention, accompanying life's moments
- If professional scene → focus on militant engagement, responsible business
- Always include the notion of "ensemble" (together) when relevant

Each message should:
1. Connect directly to the specific scene described
2. Include the militant/engaged angle of MAIF
3. Feel authentic to MAIF's caring and responsible spirit
4. Be memorable and inspiring
5. Reflect the mutualiste model

Format: Return exactly 3 messages, numbered 1-3, each on a new line.

Example adaptations:
- Family moment → "Ensemble, militants pour demain"
- Home scene → "Votre engagement, notre mission"
- Life achievement → "Militons pour vos rêves"
- Important moment → "Attention sincère, action collective"
- Daily life → "Militants du quotidien"

Generate 3 contextual messages for this scene:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message_prompt}],
            max_tokens=250,
            temperature=0.7
        )
        
        messages = response.choices[0].message.content.strip()
        # Parser les messages et enlever la numérotation
        message_list = []
        for line in messages.split('\n'):
            if line.strip() and any(line.startswith(str(i)) for i in range(1, 4)):
                # Enlever le numéro et nettoyer
                clean_message = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                clean_message = clean_message.strip('"')  # Enlever les guillemets si présents
                message_list.append(clean_message)
        
        return message_list[:3]  # S'assurer qu'on a maximum 3 messages
    
    except Exception as e:
        st.error(f"Erreur lors de la génération des messages : {str(e)}")
        return [
            "Ensemble, militants",
            "Attention sincère",
            "Militons pour demain"
        ]

def improve_prompt_with_ai(user_prompt):
    """Améliore le prompt utilisateur avec l'IA pour de meilleurs résultats"""
    try:
        improvement_prompt = f"""You are an expert prompt engineer specializing in creating prompts for photorealistic image generation. 

Take this user's basic description: "{user_prompt}"

Rewrite it as a professional photography prompt that will generate hyper-realistic, authentic lifestyle images in the style of MAIF insurance's militant and caring brand communication. 

Guidelines:
- Focus on PHOTOREALISM and documentary-style photography
- Include specific details about lighting, composition, emotions
- Mention camera settings/style (like "shot with Canon 5D Mark IV", "85mm lens", "natural lighting")
- Add authentic human details (age ranges, genuine expressions, natural poses)
- Include environmental details that feel real and lived-in
- Specify the mood and energy that fits MAIF's militant, engaged, yet caring brand
- Avoid artificial or posed-looking descriptions
- Make it sound like a brief for a professional photographer
- Focus on moments of solidarity, mutual support, sincere attention, and engagement
- Include the spirit of "société à mission" - company with purpose
- Reflect mutualiste values: community, shared responsibility, collective action
- Show authentic human connections and solidarity

Return ONLY the improved prompt, nothing else."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": improvement_prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Erreur lors de l'amélioration du prompt : {str(e)}")
        return user_prompt

def generate_image_openai(prompt, image_format="1024x1024"):
    """Génère une image avec OpenAI GPT-Image-1"""
    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=image_format,
            n=1,
        )
        
        image_data = response.data[0].b64_json
        os.makedirs("static/generated_images", exist_ok=True)
        filename = f"static/generated_images/image_{int(time.time())}.png"
        
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img.save(filename)
        
        return img
        
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'image : {str(e)}")
        return None

def add_logo_to_image(image, logo_path, position="top-left", size=(200, 115)):
    """Ajoute le logo MAIF à l'image"""
    try:
        # Ouvrir le logo
        logo = Image.open(logo_path)
        
        # Redimensionner le logo
        logo = logo.resize(size, Image.Resampling.LANCZOS)
        
        # Créer une copie de l'image
        img_copy = image.copy()
        
        # Calculer la position
        if position == "top-left":
            x, y = 20, 20
        elif position == "top-right":
            x, y = img_copy.width - logo.width - 20, 20
        elif position == "bottom-left":
            x, y = 20, img_copy.height - logo.height - 20
        else:  # bottom-right
            x, y = img_copy.width - logo.width - 20, img_copy.height - logo.height - 20
        
        # Coller le logo sur l'image
        if logo.mode == 'RGBA':
            img_copy.paste(logo, (x, y), logo)
        else:
            img_copy.paste(logo, (x, y))
        
        return img_copy
        
    except Exception as e:
        st.error(f"Erreur lors de l'ajout du logo : {str(e)}")
        return image

def add_text_overlay(image, text, position, font_size=60, rect_width_custom=None, rect_height_custom=None):
    """Ajoute du texte avec fond rouge MAIF IMPOSANT sur l'image"""
    if not text.strip():
        return image
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    using_default_font = False

    try:
        # Essayer plusieurs polices courantes sur différents systèmes
        font_paths = [
            "arial.ttf",  # Windows
            "Arial.ttf",  # Windows (variante)
            "Arial Bold.ttf",  # Windows Bold
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux
            "/usr/share/fonts/TTF/arial.ttf",  # Linux (certaines distributions)
        ]
        
        font = None
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        # Si aucune police TrueType n'est trouvée, utiliser la police par défaut MAIS avec une taille forcée
    
            
        if font is None:
            font = ImageFont.load_default()
            using_default_font = True
                
    except Exception as e:
        font = ImageFont.load_default()
        using_default_font = True

        
    # COMPENSATION pour la police par défaut
    if using_default_font:
    # Multiplier la taille pour compenser la petitesse de la police par défaut
        font_size = int(font_size * 2.5)  # Augmentation significative
    
    # Couleurs MAIF - Rouge principal selon l'identité 2019
    maif_red = "#D32F2F"  # Rouge MAIF triangle dynamique
    white_color = "#FFFFFF"
    
    # Gérer les retours à la ligne manuels dans le texte
    if '\n' in text:
        # L'utilisateur a défini ses propres lignes
        lines = text.split('\n')
    else:
        # Découpage automatique en fonction de la largeur
        words = text.split()
        lines = []
        current_line = []
        max_width = rect_width_custom if rect_width_custom else 800
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width - 120:  # Laisser de la place pour le padding
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
    
    # Calculer les dimensions du rectangle avec compensation pour police par défaut
    if using_default_font:
        line_height = int(font_size * 2.0)  # Plus grand pour police par défaut
        padding_vertical = max(80, int(font_size * 1.2))  # Plus grand
        padding_horizontal = max(100, int(font_size * 1.4))  # Plus grand
    else:
        line_height = int(font_size * 1.4)
        padding_vertical = max(40, int(font_size * 0.8))
        padding_horizontal = max(60, int(font_size * 1.0))
    
    # Utiliser les dimensions personnalisées ou calculer automatiquement
    if rect_width_custom and rect_height_custom:
        rect_width = rect_width_custom
        rect_height = rect_height_custom
    else:
        # Calcul automatique basé sur le texte
        total_height = len(lines) * line_height + padding_vertical * 2
        max_line_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines]) if lines else 200
        rect_width = max_line_width + padding_horizontal * 2
        rect_height = total_height
    
    x, y = position
    
    # S'assurer que le rectangle reste dans l'image
    if x + rect_width > img_copy.width:
        x = img_copy.width - rect_width
    if y + rect_height > img_copy.height:
        y = img_copy.height - rect_height
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    
    # Dessiner le rectangle rouge MAIF de fond IMPOSANT
    draw.rectangle([x, y, x + rect_width, y + rect_height], 
                  fill=maif_red, outline=maif_red)
    
    # Centrer le texte dans le rectangle
    total_text_height = len(lines) * line_height
    start_y = y + (rect_height - total_text_height) // 2
    
    # Dessiner chaque ligne de texte centrée
    for i, line in enumerate(lines):
        # Calculer la largeur de la ligne pour centrer horizontalement
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        text_x = x + (rect_width - line_width) // 2
        text_y = start_y + i * line_height
        
        draw.text((text_x, text_y), line, fill=white_color, font=font)
    
    return img_copy

def main():
    st.set_page_config(
        page_title="MAIF CREATIVE CREATIVE STUDIO",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS adapté aux couleurs MAIF selon Design System
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Variables CSS pour MAIF selon Design System */
    :root {
        --maif-red: #D32F2F;
        --maif-dark-red: #B71C1C;
        --maif-blue: #1976D2;
        --maif-light-blue: #E3F2FD;
        --maif-white: #FFFFFF;
        --maif-gray: #F5F5F5;
        --maif-dark-gray: #424242;
        --maif-light-gray: #FAFAFA;
        --maif-shadow: rgba(211, 47, 47, 0.15);
        --maif-shadow-deep: rgba(211, 47, 47, 0.25);
    }
    
    /* Reset et base */
    * {
        box-sizing: border-box;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--maif-gray) 0%, var(--maif-white) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--maif-dark-gray);
    }
    
    /* Header MAIF selon identité visuelle */
    .maif-hero {
        background: linear-gradient(135deg, var(--maif-red) 0%, var(--maif-dark-red) 50%, var(--maif-red) 100%);
        height: 35vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    .maif-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
        animation: shimmer 8s ease-in-out infinite alternate;
    }
    
    @keyframes shimmer {
        0% { opacity: 0.3; }
        100% { opacity: 0.7; }
    }
    
    .maif-logo {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--maif-white);
        letter-spacing: 6px;
        text-align: center;
        margin: 0;
        position: relative;
        z-index: 2;
        text-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
    }
    
    .maif-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 400;
        color: var(--maif-white);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 1rem;
        position: relative;
        z-index: 2;
        opacity: 0.9;
    }
    
    .maif-divider {
        width: 100px;
        height: 1px;
        background: var(--maif-white);
        margin: 1.5rem auto;
        position: relative;
        z-index: 2;
        opacity: 0.8;
    }
    
    /* Container principal */
    .maif-container {
        max-width: 1600px;
        margin: -80px auto 0;
        padding: 0 3rem 6rem;
        position: relative;
        z-index: 10;
    }
    
    /* Cards MAIF */
    .maif-card {
        background: var(--maif-white);
        border-radius: 0;
        box-shadow: 
            0 10px 40px var(--maif-shadow),
            0 2px 8px rgba(0, 0, 0, 0.05);
        padding: 4rem;
        margin: 3rem 0;
        border: 1px solid rgba(211, 47, 47, 0.1);
        position: relative;
        transition: all 0.6s cubic-bezier(0.23, 1, 0.32, 1);
    }
    
    .maif-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--maif-red) 50%, transparent 100%);
        opacity: 0;
        transition: opacity 0.6s ease;
    }
    
    .maif-card:hover::before {
        opacity: 1;
    }
    
    .maif-card:hover {
        transform: translateY(-12px);
        box-shadow: 
            0 25px 60px var(--maif-shadow-deep),
            0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Card résultat MAIF */
    .maif-result-card {
        background: linear-gradient(135deg, var(--maif-red) 0%, var(--maif-dark-red) 100%);
        color: var(--maif-white);
        position: relative;
        overflow: hidden;
    }
    
    .maif-result-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--maif-white) 0%, rgba(255,255,255,0.8) 50%, var(--maif-white) 100%);
    }
    
    /* Titres de section selon Design System */
    .maif-section-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--maif-dark-gray);
        margin: 3rem 0 2rem 0;
        text-align: center;
        position: relative;
        letter-spacing: 1px;
    }
    
    .maif-section-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 1px;
        background: var(--maif-red);
    }
    
    /* Numérotation MAIF */
    .maif-section-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 2px solid var(--maif-red);
        border-radius: 50%;
        text-align: center;
        line-height: 36px;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: var(--maif-red);
        margin-right: 1rem;
        font-size: 1.1rem;
    }
    
    /* Inputs selon accessibilité MAIF */
    .stTextArea textarea {
        border: 1px solid rgba(211, 47, 47, 0.3);
        border-radius: 0;
        padding: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        line-height: 1.6;
        background: var(--maif-light-gray);
        transition: all 0.4s ease;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--maif-red);
        box-shadow: 0 0 0 2px rgba(211, 47, 47, 0.1);
        background: var(--maif-white);
        outline: none;
    }
    
    /* Boutons MAIF selon Design System */
    .stButton button {
        background: linear-gradient(135deg, var(--maif-red) 0%, var(--maif-dark-red) 100%);
        color: var(--maif-white);
        border: 1px solid var(--maif-red);
        border-radius: 0;
        padding: 1rem 2.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(211, 47, 47, 0.3);
        border-color: var(--maif-white);
    }
    
    /* Boutons primaires */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--maif-red) 0%, var(--maif-dark-red) 100%) !important;
        color: var(--maif-white) !important;
        border-color: var(--maif-red) !important;
        font-weight: 700 !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--maif-dark-red) 0%, var(--maif-red) 100%) !important;
        box-shadow: 0 8px 25px rgba(211, 47, 47, 0.4) !important;
    }
    
    /* Boutons secondaires */
    .stButton button[kind="secondary"] {
        background: transparent !important;
        color: var(--maif-red) !important;
        border: 2px solid var(--maif-red) !important;
        padding: 0.8rem 2rem !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background: var(--maif-red) !important;
        color: var(--maif-white) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border: 1px solid rgba(211, 47, 47, 0.3);
        border-radius: 0;
        background: var(--maif-light-gray);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--maif-red);
        box-shadow: 0 0 0 2px rgba(211, 47, 47, 0.1);
    }
    
    /* Cards de format */
    .format-card {
        border: 2px solid rgba(211, 47, 47, 0.3);
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        background: var(--maif-white);
        position: relative;
    }
    
    .format-card:hover {
        border-color: var(--maif-red);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px var(--maif-shadow);
    }
    
    .format-card.selected {
        border-color: var(--maif-red);
        background: linear-gradient(135deg, var(--maif-gray) 0%, var(--maif-white) 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(211, 47, 47, 0.2);
    }
    
    .format-card.selected::before {
        content: '✓';
        position: absolute;
        top: 8px;
        right: 8px;
        color: var(--maif-red);
        font-weight: bold;
        font-size: 1rem;
    }
    
    .format-preview {
        width: 80px;
        height: 60px;
        border: 2px solid #ddd;
        margin: 0 auto 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        color: #666;
        background: #f9f9f9;
    }
    
    .format-name {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: var(--maif-dark-gray);
        margin-bottom: 0.3rem;
    }
    
    .format-dimensions {
        font-size: 0.75rem;
        color: #666;
    }
    
    /* Palettes de couleurs MAIF selon Design System */
    .maif-palette-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
        max-width: 100%;
    }

    .maif-color-palette {
        background: var(--maif-white);
        border: 1px solid rgba(211, 47, 47, 0.2);
        padding: 0.8rem;
        text-align: center;
        transition: all 0.4s ease;
        cursor: pointer;
        position: relative;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .maif-color-palette:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px var(--maif-shadow);
        border-color: var(--maif-red);
    }

    .maif-color-palette.selected {
        border: 2px solid var(--maif-red);
        background: linear-gradient(135deg, var(--maif-gray) 0%, var(--maif-white) 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(211, 47, 47, 0.2);
    }

    .maif-color-palette.selected::before {
        content: '✓';
        position: absolute;
        top: 8px;
        right: 8px;
        color: var(--maif-red);
        font-weight: bold;
        font-size: 1rem;
    }

    .color-swatches {
        display: flex;
        justify-content: center;
        gap: 3px;
        margin: 0.5rem 0;
    }

    .color-swatch {
        width: 25px;
        height: 25px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        flex-shrink: 0;
    }

    .palette-name {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--maif-dark-gray);
        margin-bottom: 0.3rem;
    }

    .palette-description {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
        line-height: 1.2;
    }
    
    /* Prompts prédéfinis selon valeurs MAIF */
    .maif-prompt-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .maif-prompt-card {
        background: linear-gradient(135deg, var(--maif-light-gray) 0%, var(--maif-white) 100%);
        border: 1px solid rgba(211, 47, 47, 0.2);
        padding: 1.5rem;
        text-align: left;
        transition: all 0.4s ease;
        cursor: pointer;
        position: relative;
    }
    
    .maif-prompt-card:hover {
        border-color: var(--maif-red);
        background: var(--maif-white);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px var(--maif-shadow);
    }
    
    .prompt-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: var(--maif-red);
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .prompt-text {
        font-size: 0.9rem;
        line-height: 1.4;
        color: #555;
    }
    
    /* Messages de marque militant */
    .maif-message-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .maif-message-card {
        background: linear-gradient(135deg, #f8f8f8 0%, var(--maif-white) 100%);
        border: 1px solid rgba(211, 47, 47, 0.3);
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        font-family: 'Source Sans Pro', sans-serif;
        font-style: italic;
        font-weight: 500;
    }
    
    .maif-message-card:hover {
        background: var(--maif-red);
        color: var(--maif-white);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(211, 47, 47, 0.3);
    }
    
    /* Contrôles selon Design System */
    .stSlider > div > div > div {
        background: var(--maif-red) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--maif-dark-gray) !important;
    }
    
    .stNumberInput input {
        border: 1px solid rgba(211, 47, 47, 0.3);
        border-radius: 0;
        background: var(--maif-light-gray);
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus {
        border-color: var(--maif-red);
        box-shadow: 0 0 0 2px rgba(211, 47, 47, 0.1);
        background: var(--maif-white);
        outline: none;
    }
    
    .stCheckbox > label {
        font-family: 'Inter', sans-serif;
        color: var(--maif-dark-gray);
        font-weight: 500;
    }
    
    .stCheckbox input:checked + div {
        background-color: var(--maif-red) !important;
        border-color: var(--maif-red) !important;
    }
    
    /* Zone d'amélioration du prompt */
    .maif-enhanced-prompt {
        background: linear-gradient(135deg, #fefefe 0%, var(--maif-gray) 100%);
        border: 2px solid var(--maif-red);
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
    }
    
    .maif-enhanced-prompt::before {
        content: 'ENHANCED';
        position: absolute;
        top: -10px;
        left: 20px;
        background: var(--maif-red);
        color: var(--maif-white);
        padding: 0.3rem 1rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Zone de résultat */
    .maif-result-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--maif-white);
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 2px;
    }
    
    .maif-placeholder {
        text-align: center;
        padding: 6rem 2rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .maif-placeholder h3 {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
    }
    
    .maif-placeholder p {
        font-size: 1rem;
        font-weight: 300;
    }
    
    /* Tips section */
    .maif-tips {
        border: 2px solid var(--maif-red);
        background: linear-gradient(135deg, var(--maif-gray) 0%, var(--maif-white) 100%);
        padding: 2rem;
        margin: 3rem 0;
        position: relative;
    }
    
    .maif-tips::before {
        content: 'STUDIO TIPS';
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--maif-white);
        color: var(--maif-red);
        padding: 0.5rem 1.5rem;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    .tips-title {
        font-family: 'Source Sans Pro', sans-serif;
        color: var(--maif-dark-gray);
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .tips-content {
        color: #444;
        line-height: 1.8;
        font-size: 0.95rem;
    }
    
    /* Palettes compactes */
    .maif-color-palette-compact {
        border: 1px solid rgba(211, 47, 47, 0.2);
        padding: 0.5rem;
        text-align: center;
        transition: all 0.4s ease;
        cursor: pointer;
        position: relative;
        background: var(--maif-white);
    }

    .maif-color-palette-compact:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px var(--maif-shadow);
        border-color: var(--maif-red);
    }

    .maif-color-palette-compact.selected {
        border: 2px solid var(--maif-red);
        background: linear-gradient(135deg, var(--maif-gray) 0%, var(--maif-white) 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(211, 47, 47, 0.2);
    }

    .maif-color-palette-compact.selected::before {
        content: '✓';
        position: absolute;
        top: 5px;
        right: 5px;
        color: var(--maif-red);
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    /* Masquer les éléments Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .maif-logo {
            font-size: 2.5rem;
            letter-spacing: 4px;
        }
        
        .maif-container {
            margin-top: -60px;
            padding: 0 1.5rem 3rem;
        }
        
        .maif-card {
            padding: 2rem;
        }
        
        .maif-palette-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.8rem;
        }
        
        .maif-prompt-grid {
            grid-template-columns: 1fr;
        }
    }
    
    @media (max-width: 480px) {
        .maif-palette-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section MAIF selon identité 2019
    st.markdown("""
    <div class="maif-hero">
        <h1 class="maif-logo">MAIF </h1>
        <div class="maif-divider"></div>
        <p class="maif-subtitle"> CREATIVE STUDIO</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Container principal
    st.markdown('<div class="maif-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        st.markdown('<div class="maif-card">', unsafe_allow_html=True)
        
        # Section 0: Format d'image
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">0</span>
            Image Format
        </div>
        """, unsafe_allow_html=True)
        
        # Définir les formats d'image
        image_formats = {
            "Square": {
                "size": "1024x1024",
                "ratio": "1:1",
                "description": "Perfect for Instagram posts",
                "preview_ratio": 1.0
            },
            "Portrait": {
                "size": "1024x1536",
                "ratio": "2:3",
                "description": "Instagram Stories, vertical posts",
                "preview_ratio": 1.5
            },
            "Landscape": {
                "size": "1536x1024",
                "ratio": "3:2",
                "description": "LinkedIn, horizontal posts",
                "preview_ratio": 0.67
            }
        }
        
        selected_format = st.session_state.get('selected_format', "Square")
        
        # Affichage des formats en 3 colonnes
        col_format1, col_format2, col_format3 = st.columns(3, gap="small")
        
        with col_format1:
            format_data = image_formats["Square"]
            selected_class = "selected" if selected_format == "Square" else ""
            
            format_html = f"""
            <div class="format-card {selected_class}" style="border: 2px solid rgba(211, 47, 47, 0.3); padding: 1rem; text-align: center; background: white; margin-bottom: 0.5rem;">
                <div style="width: 60px; height: 60px; border: 2px solid #ddd; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #666; background: #f9f9f9;">
                    {format_data["ratio"]}
                </div>
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; margin-bottom: 0.3rem;">Square</div>
                <div style="font-size: 0.75rem; color: #666; margin-bottom: 0.3rem;">{format_data["size"]}</div>
                <div style="font-size: 0.7rem; color: #888; font-style: italic;">{format_data["description"]}</div>
            </div>
            """
            st.markdown(format_html, unsafe_allow_html=True)
            
            if st.button("SELECT SQUARE", key="format_Square", 
                       type="primary" if selected_format == "Square" else "secondary",
                       use_container_width=True):
                st.session_state.selected_format = "Square"
                st.rerun()
        
        with col_format2:
            format_data = image_formats["Portrait"]
            selected_class = "selected" if selected_format == "Portrait" else ""
            
            format_html = f"""
            <div class="format-card {selected_class}" style="border: 2px solid rgba(211, 47, 47, 0.3); padding: 1rem; text-align: center; background: white; margin-bottom: 0.5rem;">
                <div style="width: 40px; height: 60px; border: 2px solid #ddd; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #666; background: #f9f9f9;">
                    {format_data["ratio"]}
                </div>
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; margin-bottom: 0.3rem;">Portrait</div>
                <div style="font-size: 0.75rem; color: #666; margin-bottom: 0.3rem;">{format_data["size"]}</div>
                <div style="font-size: 0.7rem; color: #888; font-style: italic;">{format_data["description"]}</div>
            </div>
            """
            st.markdown(format_html, unsafe_allow_html=True)
            
            if st.button("SELECT PORTRAIT", key="format_Portrait", 
                       type="primary" if selected_format == "Portrait" else "secondary",
                       use_container_width=True):
                st.session_state.selected_format = "Portrait"
                st.rerun()
        
        with col_format3:
            format_data = image_formats["Landscape"]
            selected_class = "selected" if selected_format == "Landscape" else ""
            
            format_html = f"""
            <div class="format-card {selected_class}" style="border: 2px solid rgba(211, 47, 47, 0.3); padding: 1rem; text-align: center; background: white; margin-bottom: 0.5rem;">
                <div style="width: 90px; height: 60px; border: 2px solid #ddd; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #666; background: #f9f9f9;">
                    {format_data["ratio"]}
                </div>
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; margin-bottom: 0.3rem;">Landscape</div>
                <div style="font-size: 0.75rem; color: #666; margin-bottom: 0.3rem;">{format_data["size"]}</div>
                <div style="font-size: 0.7rem; color: #888; font-style: italic;">{format_data["description"]}</div>
            </div>
            """
            st.markdown(format_html, unsafe_allow_html=True)
            
            if st.button("SELECT LANDSCAPE", key="format_Landscape", 
                       type="primary" if selected_format == "Landscape" else "secondary",
                       use_container_width=True):
                st.session_state.selected_format = "Landscape"
                st.rerun()
        
        # Section 1: Palette de couleurs MAIF selon Design System
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">I</span>
            Color Palette
        </div>
        """, unsafe_allow_html=True)
        
        # Définir les palettes de couleurs MAIF selon Design System
        color_palettes = {
            "MAIF Rouge": {
                "colors": ["#D32F2F", "#F44336", "#EF5350", "#FFCDD2"],
                "description": "Rouge triangle dynamique",
                "prompt": "dynamic red lighting with MAIF triangle colors and militant engagement tones"
            },
            "Bleu Confiance": {
                "colors": ["#1976D2", "#2196F3", "#64B5F6", "#E3F2FD"],
                "description": "Confiance et sécurité",
                "prompt": "trustworthy blue lighting with confidence and security professional tones"
            },
            "Naturel Solidaire": {
                "colors": ["#F5F5DC", "#DEB887", "#D2B48C", "#BC9A6A"],
                "description": "Tons solidaires naturels",
                "prompt": "warm natural lighting with solidarity and community tones"
            },
            "Engagement Vert": {
                "colors": ["#388E3C", "#4CAF50", "#81C784", "#C8E6C9"],
                "description": "Transition écologique",
                "prompt": "engaged green lighting with ecological transition and sustainability tones"
            },
            "Militant Orange": {
                "colors": ["#F57C00", "#FF9800", "#FFB74D", "#FFE0B2"],
                "description": "Énergie militante",
                "prompt": "militant orange lighting with activism and energy tones"
            },
            "Société Violette": {
                "colors": ["#7B1FA2", "#9C27B0", "#BA68C8", "#E1BEE7"],
                "description": "Innovation sociale",
                "prompt": "innovative purple lighting with social mission and innovation tones"
            }
        }
        
        selected_palette = st.session_state.get('selected_palette', None)
        palette_names = list(color_palettes.keys())
        
        # PREMIÈRE LIGNE : MAIF Rouge, Bleu Confiance, Naturel Solidaire
        col1_row1, col2_row1, col3_row1 = st.columns(3, gap="small")
        
        with col1_row1:
            # MAIF Rouge palette
            palette_data = color_palettes["MAIF Rouge"]
            selected_class = "selected" if selected_palette == "MAIF Rouge" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">MAIF Rouge</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT MAIF ROUGE", key="palette_MAIF Rouge", 
                       type="primary" if selected_palette == "MAIF Rouge" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "MAIF Rouge" if selected_palette != "MAIF Rouge" else None
                st.rerun()
        
        with col2_row1:
            # Bleu Confiance palette
            palette_data = color_palettes["Bleu Confiance"]
            selected_class = "selected" if selected_palette == "Bleu Confiance" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Bleu Confiance</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT BLEU CONFIANCE", key="palette_Bleu Confiance", 
                       type="primary" if selected_palette == "Bleu Confiance" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "Bleu Confiance" if selected_palette != "Bleu Confiance" else None
                st.rerun()
        
        with col3_row1:
            # Naturel Solidaire palette
            palette_data = color_palettes["Naturel Solidaire"]
            selected_class = "selected" if selected_palette == "Naturel Solidaire" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Naturel Solidaire</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT NATUREL SOLIDAIRE", key="palette_Naturel Solidaire", 
                       type="primary" if selected_palette == "Naturel Solidaire" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "Naturel Solidaire" if selected_palette != "Naturel Solidaire" else None
                st.rerun()
        
        # DEUXIÈME LIGNE : Engagement Vert, Militant Orange, Société Violette
        col1_row2, col2_row2, col3_row2 = st.columns(3, gap="small")
        
        with col1_row2:
            # Engagement Vert palette
            palette_data = color_palettes["Engagement Vert"]
            selected_class = "selected" if selected_palette == "Engagement Vert" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Engagement Vert</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT ENGAGEMENT VERT", key="palette_Engagement Vert", 
                       type="primary" if selected_palette == "Engagement Vert" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "Engagement Vert" if selected_palette != "Engagement Vert" else None
                st.rerun()
        
        with col2_row2:
            # Militant Orange palette
            palette_data = color_palettes["Militant Orange"]
            selected_class = "selected" if selected_palette == "Militant Orange" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Militant Orange</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT MILITANT ORANGE", key="palette_Militant Orange", 
                       type="primary" if selected_palette == "Militant Orange" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "Militant Orange" if selected_palette != "Militant Orange" else None
                st.rerun()
        
        with col3_row2:
            # Société Violette palette
            palette_data = color_palettes["Société Violette"]
            selected_class = "selected" if selected_palette == "Société Violette" else ""
            
            palette_html = f"""
            <div class="maif-color-palette-compact {selected_class}" style="text-align: center; padding: 0.8rem; border: 1px solid rgba(211, 47, 47, 0.2); background: white; margin-bottom: 0.5rem;">
                <div style="font-family: 'Source Sans Pro', serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Société Violette</div>
                <div style="display: flex; justify-content: center; gap: 3px; margin: 0.5rem 0;">
            """
            for color in palette_data["colors"]:
                palette_html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid rgba(0,0,0,0.1);"></div>'
            
            palette_html += f"""
                </div>
                <div style="font-size: 0.7rem; color: #666; font-style: italic;">{palette_data["description"]}</div>
            </div>
            """
            st.markdown(palette_html, unsafe_allow_html=True)
            
            if st.button("SELECT SOCIÉTÉ VIOLETTE", key="palette_Société Violette", 
                       type="primary" if selected_palette == "Société Violette" else "secondary",
                       use_container_width=True):
                st.session_state.selected_palette = "Société Violette" if selected_palette != "Société Violette" else None
                st.rerun()
        
        # Section 2: Description de scène MAIF selon valeurs mutualistes
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">II</span>
            Scene Composition
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Ready-to-Use Inspirations Militantes:**")

        default_prompts = {
            "Communauté Solidaire": "Groupe de sociétaires MAIF se réunissant pour un projet commun, sentiment de solidarité mutualiste, expressions d'engagement collectif et de responsabilité partagée",
            "Famille Militante": "Famille engagée dans des actions citoyennes, enfants apprenant les valeurs de solidarité, ambiance de transmission des valeurs MAIF", 
            "Entrepreneur Responsable": "Dirigeant d'entreprise socialement responsable dans son bureau, regard déterminé vers l'avenir durable, esprit de société à mission",
            "Transition Écologique": "Jeunes militants pour l'environnement plantant des arbres, énergie positive de changement, engagement pour la transition écologique",
            "Assemblée Démocratique": "Représentants des sociétaires MAIF en assemblée générale, débat démocratique et participation citoyenne, gouvernance participative",
            "Innovation Sociale": "Équipe diversifiée travaillant sur des projets d'innovation sociale, collaboration intergénérationnelle, esprit d'entreprise à mission"
        }

        # PREMIÈRE LIGNE : Communauté Solidaire, Famille Militante, Entrepreneur Responsable
        col1_prompt1, col2_prompt1, col3_prompt1 = st.columns(3, gap="small")

        with col1_prompt1:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Communauté Solidaire</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Communauté Solidaire"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE COMMUNAUTÉ SOLIDAIRE", key="default_prompt_0", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Communauté Solidaire"]
                st.rerun()

        with col2_prompt1:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Famille Militante</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Famille Militante"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE FAMILLE MILITANTE", key="default_prompt_1", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Famille Militante"]
                st.rerun()

        with col3_prompt1:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Entrepreneur Responsable</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Entrepreneur Responsable"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE ENTREPRENEUR RESPONSABLE", key="default_prompt_2", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Entrepreneur Responsable"]
                st.rerun()

        # DEUXIÈME LIGNE : Transition Écologique, Assemblée Démocratique, Innovation Sociale
        col1_prompt2, col2_prompt2, col3_prompt2 = st.columns(3, gap="small")

        with col1_prompt2:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Transition Écologique</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Transition Écologique"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE TRANSITION ÉCOLOGIQUE", key="default_prompt_3", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Transition Écologique"]
                st.rerun()

        with col2_prompt2:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Assemblée Démocratique</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Assemblée Démocratique"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE ASSEMBLÉE DÉMOCRATIQUE", key="default_prompt_4", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Assemblée Démocratique"]
                st.rerun()

        with col3_prompt2:
            prompt_html = f"""
            <div class="maif-prompt-card" style="background: linear-gradient(135deg, #FAFAFA 0%, white 100%); border: 1px solid rgba(211, 47, 47, 0.2); padding: 1.2rem; text-align: left; margin-bottom: 0.5rem; min-height: 120px;">
                <div style="font-family: 'Source Sans Pro', serif; font-weight: 600; color: #D32F2F; font-size: 1rem; margin-bottom: 0.5rem;">Innovation Sociale</div>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #555;">{default_prompts["Innovation Sociale"]}</div>
            </div>
            """
            st.markdown(prompt_html, unsafe_allow_html=True)
            
            if st.button("USE INNOVATION SOCIALE", key="default_prompt_5", use_container_width=True):
                st.session_state.selected_default_prompt = default_prompts["Innovation Sociale"]
                st.rerun()
        
        # Prompt de base MAIF selon raison d'être
        base_prompt = """Créez une photographie hyper-réaliste et authentique dans le style de la communication MAIF 'assureur militant' et société à mission. Concentrez-vous sur des détails PHOTORÉALISTES avec :

- Émotions humaines RÉELLES exprimant l'engagement, la solidarité et l'attention sincère
- Qualité photographique professionnelle avec éclairage naturel inspirant
- Personnes diverses incarnant les valeurs mutualistes : humanisme, solidarité, démocratie
- Style photojournalistique capturant des moments d'engagement collectif et de responsabilité partagée
- Détails haute résolution dans les expressions d'engagement, texture de peau, environnements collaboratifs
- Sentiment de communauté mutualiste, action collective et mieux commun
- Style documentaire 'société à mission' qui semble complètement naturel et inspirant
- Éviter l'aspect artificiel - viser le réalisme photographique militant
- Ambiance d'engagement responsable, solidarité active et attention sincère portée à l'autre et au monde

"""

        # Ajouter la palette de couleurs si sélectionnée
        if selected_palette and selected_palette in color_palettes:
            color_instruction = f"- Palette de couleurs et éclairage : {color_palettes[selected_palette]['prompt']}\n"
            base_prompt += color_instruction
        
        base_prompt += "Scène réaliste spécifique militant : "
        
        # Zone de texte pour la description
        default_scene = st.session_state.get('selected_default_prompt', '')
        user_prompt = st.text_area(
            "",
            value=default_scene,
            placeholder="Une communauté de sociétaires MAIF engagés dans une action collective, expressions d'attention sincère et de solidarité mutualiste...",
            height=120,
            key="user_prompt",
            label_visibility="collapsed"
        )
        
        # Section d'amélioration
        col_prompt1, col_prompt2 = st.columns([3, 1])
        
        with col_prompt1:
            if 'improved_prompt' in st.session_state:
                st.markdown('<div class="maif-enhanced-prompt">', unsafe_allow_html=True)
                improved_display = st.text_area(
                    "",
                    value=st.session_state.improved_prompt,
                    height=140,
                    key="improved_prompt_display",
                    label_visibility="collapsed"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                final_user_prompt = improved_display
            else:
                final_user_prompt = user_prompt
        
        with col_prompt2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.button("ENHANCE", type="secondary"):
                if user_prompt.strip():
                    with st.spinner("Raffinement militant en cours..."):
                        improved = improve_prompt_with_ai(user_prompt)
                        st.session_state.improved_prompt = improved
                        st.rerun()
                else:
                    st.warning("Please write description first")
            
            if 'improved_prompt' in st.session_state:
                if st.button("RESET"):
                    if 'improved_prompt' in st.session_state:
                        del st.session_state.improved_prompt
                    st.rerun()
        
        full_prompt = base_prompt + final_user_prompt if final_user_prompt else base_prompt + "Une communauté MAIF unie dans l'action collective, portant une attention sincère à l'autre et au monde"
        
        # Section 3: Message de marque MAIF militant
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">III</span>
            Brand Message
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour générer des suggestions
        col_msg1, col_msg2 = st.columns([3, 1])
        
        with col_msg2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("GENERATE MESSAGES", type="secondary"):
                if final_user_prompt.strip():
                    with st.spinner("Génération militante..."):
                        suggested_messages = generate_brand_messages(final_user_prompt)
                        st.session_state.suggested_messages = suggested_messages
                        st.rerun()
                else:
                    st.warning("Please describe your scene first")
        
        # Affichage des suggestions
        if 'suggested_messages' in st.session_state:
            st.markdown("**AI Suggestions Militantes:**")
            st.markdown('<div class="maif-message-grid">', unsafe_allow_html=True)
            for i, message in enumerate(st.session_state.suggested_messages):
                message_html = f"""
                <div class="maif-message-card">
                    "{message}"
                </div>
                """
                st.markdown(message_html, unsafe_allow_html=True)
                
                if st.button(f"Select Message {i+1}", key=f"msg_btn_{i}", use_container_width=True):
                    st.session_state.selected_message = message
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_msg1:
            # Zone de texte pour le message
            default_message = st.session_state.get('selected_message', '')
            overlay_text = st.text_area(
                "",
                value=default_message,
                placeholder="Ensemble, militants\n(Utilisez les retours à la ligne pour contrôler les lignes de texte)",
                height=100,
                key="overlay_text",
                label_visibility="collapsed",
                help="Astuce : Ajoutez des retours à la ligne (Entrée) pour contrôler la répartition du texte"
            )
        
        # Section 4: Positionnement du texte
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">IV</span>
            Text Positioning
        </div>
        """, unsafe_allow_html=True)
        
        col_pos1, col_pos2, col_pos3 = st.columns(3)
        
        with col_pos1:
            text_x = st.slider("Position X", 0, 800, 579, key="pos_x")
        
        with col_pos2:
            text_y = st.slider("Position Y", 0, 800, 753, key="pos_y")
        
        with col_pos3:
            font_size = st.slider("Taille Typographique", 20, 120, 60, key="font_size")
        
        # Section 5: Dimensions du rectangle
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">V</span>
            Rectangle Dimensions
        </div>
        """, unsafe_allow_html=True)
        
        col_size1, col_size2 = st.columns(2)
        
        with col_size1:
            rect_width = st.number_input(
                "Largeur", 
                min_value=100, 
                max_value=800, 
                value=740, 
                step=10,
                key="rect_width"
            )
        
        with col_size2:
            rect_height = st.number_input(
                "Hauteur", 
                min_value=50, 
                max_value=400, 
                value=190, 
                step=10,
                key="rect_height"
            )
        
        # Section 6: Options du logo
        st.markdown("""
        <div class="maif-section-title">
            <span class="maif-section-number">VI</span>
            Logo Options
        </div>
        """, unsafe_allow_html=True)
        
        col_logo1, col_logo2 = st.columns([1, 1])
        
        with col_logo1:
            add_logo = st.checkbox("Ajouter le Logo MAIF", key="add_logo")
            
            if add_logo:
                logo_position = st.selectbox(
                    "Position du Logo",
                    ["top-left", "top-right", "bottom-left", "bottom-right"],
                    index=0,
                    key="logo_position"
                )
        
        with col_logo2:
            if add_logo:
                logo_width = st.number_input(
                    "Largeur Logo", 
                    min_value=50, 
                    max_value=300, 
                    value=200, 
                    step=10,
                    key="logo_width"
                )
                
                logo_height = st.number_input(
                    "Hauteur Logo", 
                    min_value=25, 
                    max_value=150, 
                    value=115, 
                    step=5,
                    key="logo_height"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton de génération
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("CREATE IMAGE", type="primary", use_container_width=True):
            if final_user_prompt.strip():
                with st.spinner("Creating militant content..."):
                    # Récupérer le format sélectionné
                    selected_image_format = image_formats[selected_format]["size"]
                    generated_image = generate_image_openai(full_prompt, selected_image_format)
                    
                    if generated_image:
                        st.session_state.generated_image = generated_image
                        st.success("Image militante créée avec succès !")
                    else:
                        st.error("Creation failed")
            else:
                st.warning("Please describe your militant scene")
    
    with col2:
        st.markdown('<div class="maif-card maif-result-card">', unsafe_allow_html=True)
        st.markdown('<div class="maif-result-title">CRÉATION MILITANTE</div>', unsafe_allow_html=True)
        
        if 'generated_image' in st.session_state:
            # Commencer avec l'image générée
            final_image = st.session_state.generated_image
            
            # Ajouter le texte si spécifié
            if overlay_text.strip():
                final_image = add_text_overlay(
                    final_image, 
                    overlay_text, 
                    (text_x, text_y),
                    font_size,
                    rect_width,
                    rect_height
                )
            
            # Ajouter le logo si demandé
            if add_logo:
                logo_path = "MAIF.png"  # Vous devrez ajouter le logo MAIF triangle rouge
                logo_size = (logo_width, logo_height) if add_logo else (200, 115)
                final_image = add_logo_to_image(
                    final_image, 
                    logo_path, 
                    logo_position if add_logo else "top-left",
                    logo_size
                )
            
            st.image(final_image, use_container_width=True)
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                buf = BytesIO()
                final_image.save(buf, format="PNG")
                byte_data = buf.getvalue()
                
                # Nom de fichier avec format
                format_name = selected_format.lower()
                st.download_button(
                    label="DOWNLOAD",
                    data=byte_data,
                    file_name=f"maif_creation_{format_name}_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_btn2:
                if st.button("NEW IMAGE", use_container_width=True):
                    # Relancer une nouvelle génération avec le même prompt
                    if 'generated_image' in st.session_state:
                        del st.session_state.generated_image
                    
                    # Récupérer le prompt actuel pour regénérer
                    current_prompt = st.session_state.get('improved_prompt_display', st.session_state.get('user_prompt', ''))
                    if current_prompt.strip():
                        with st.spinner("Nouvelle création ..."):
                            # Utiliser le prompt complet comme dans la génération principale
                            if 'improved_prompt' in st.session_state:
                                final_prompt = base_prompt + st.session_state.improved_prompt
                            else:
                                final_prompt = base_prompt + st.session_state.get('user_prompt', '')
                            
                            # Récupérer le format sélectionné
                            selected_image_format = image_formats[selected_format]["size"]
                            new_image = generate_image_openai(final_prompt, selected_image_format)
                            if new_image:
                                st.session_state.generated_image = new_image
                                st.rerun()
                    else:
                        st.warning("Aucun prompt disponible pour la régénération")
        
        else:
            st.markdown("""
            <div class="maif-placeholder">
                <h3>Atelier Militant</h3>
                <p>Configurez votre composition engagée et lancez la création</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Section conseils MAIF militants
            st.markdown("""
            <div class="maif-tips">
                <div class="tips-title">Studio Tips Militants:</div>
                <div class="tips-content">
                    • Choose your format first: Square (1:1) for Instagram, Portrait (2:3) for Stories, Landscape (3:2) for LinkedIn<br>
                    • Focus on engagement, solidarity and collective action<br>
                    • Use 'militant', 'engaged', 'community', 'solidarity'<br>
                    • Describe specific activist scenes (assemblies, collective projects, social innovation)<br>
                    • Add 'documentary style' or 'photojournalistic approach'<br>
                    • Prioritize authentic engagement over perfection<br>
                    • Reflect MAIF's mutualiste values and société à mission spirit
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
