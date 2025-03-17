import streamlit as st
import tempfile
import os
from gtts import gTTS
import base64

def text_to_speech(text, language='pt-br', slow=False):
    """
    Converte texto para fala usando a biblioteca gTTS
    """
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        return tts
    except Exception as e:
        st.error(f"Erro ao converter texto para fala: {e}")
        return None

def get_audio_download_link(audio_file_path, filename="audio.mp3"):
    """
    Gera um link para download do arquivo de áudio
    """
    with open(audio_file_path, "rb") as file:
        audio_bytes = file.read()
    
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Baixar arquivo de áudio</a>'
    return href

def process_file(file, slow_speed):
    """
    Processa o arquivo enviado pelo usuário
    """
    try:
        # Cria um arquivo temporário para salvar o conteúdo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp:
            temp.write(file.getvalue())
            temp_path = temp.name
        
        # Lê o conteúdo do arquivo
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove o arquivo temporário
        os.unlink(temp_path)
        
        # Exibe o conteúdo do arquivo
        st.subheader("Conteúdo do arquivo:")
        st.text_area("Texto extraído:", value=content, height=250)
        
        # Converte texto para fala
        tts = text_to_speech(content, slow=slow_speed)
        
        if tts:
            # Cria um arquivo temporário para o áudio
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            audio_file_path = audio_file.name
            audio_file.close()
            
            # Salva o áudio no arquivo temporário
            tts.save(audio_file_path)
            
            # Exibe o player de áudio
            st.subheader("Áudio gerado:")
            st.audio(audio_file_path)
            
            # Oferece opção para baixar o arquivo
            st.markdown(get_audio_download_link(audio_file_path, f"{file.name}.mp3"), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

def main():
    st.set_page_config(
        page_title="Conversor de Texto para Fala - Português Brasil",
        page_icon="🔊",
        layout="wide"
    )
    
    st.title("🎙️ Conversor de Texto para Fala - Português Brasil")
    st.write("""
    Este aplicativo converte arquivos de texto para áudio usando síntese de voz em português do Brasil.
    Basta fazer o upload de um arquivo de texto (.txt) e o aplicativo irá gerar um arquivo de áudio para você.
    """)
    
    # Interface para upload de arquivo
    uploaded_file = st.file_uploader("Faça upload do arquivo de texto", type=["txt"])
    
    # Opção para velocidade de fala
    slow_speed = st.checkbox("Velocidade de fala mais lenta")
    
    if uploaded_file is not None:
        # Processa o arquivo
        process_file(uploaded_file, slow_speed)
        
    # Opção alternativa de entrada direta de texto
    st.subheader("Ou digite o texto diretamente:")
    direct_text = st.text_area("Digite seu texto aqui:", height=150)
    
    if st.button("Converter para fala") and direct_text:
        tts = text_to_speech(direct_text, slow=slow_speed)
        
        if tts:
            # Cria um arquivo temporário para o áudio
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            audio_file_path = audio_file.name
            audio_file.close()
            
            # Salva o áudio no arquivo temporário
            tts.save(audio_file_path)
            
            # Exibe o player de áudio
            st.subheader("Áudio gerado:")
            st.audio(audio_file_path)
            
            # Oferece opção para baixar o arquivo
            st.markdown(get_audio_download_link(audio_file_path, "texto_direto.mp3"), unsafe_allow_html=True)
    
    # Rodapé
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ usando Streamlit e gTTS")

if __name__ == "__main__":
    main()
