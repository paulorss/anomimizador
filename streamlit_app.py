import streamlit as st
import base64
import io
import re
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader
import spacy
import regex as re
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configuração da página
st.set_page_config(
    page_title="Anonimizador de Textos - LGPD",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="auto",
)

# Função simples de anonimização que não depende do Presidio
def anonimizar_simples(texto, palavras_adicionais=None, mascara='*'):
    """
    Função de anonimização simples usando expressões regulares
    """
    if not texto:
        return texto, []
    
    # Lista para armazenar os achados
    achados = []
    texto_original = texto
    
    # Função auxiliar para substituir mantendo o case original
    def substituir_preservando_maiusculas(match):
        trecho = match.group(0)
        return mascara * len(trecho)
        
    # Lista de termos sensíveis a serem sempre anonimizados
    termos_sensiveis = [
        # Raça e etnia
        "Raça", "etnia", "cor da pele", "origem racial", "afrodescendente", 
        "indígena", "branco", "negro", "pardo", "amarelo", "ascendência", 
        "nacionalidade",
        
        # Religião
        "religião", "crença", "fé", "igreja", "templo", "culto", "católico", 
        "evangélico", "protestante", "espírita", "candomblé", "umbanda", "ateu", 
        "agnóstico", "judaísmo", "islamismo", "budismo",
        
        # Opinião política
        "opinião política", "partido político", "filiação partidária", "esquerda", 
        "direita", "centro", "conservador", "progressista", "liberal", "sindicalista", 
        "sindicato", "filiação sindical", "sindicalizado",
        
        # Orientação sexual
        "orientação sexual", "heterossexual", "homossexual", "bissexual", "gay", "casado", "solteiro", "víuva", "casada", "divorciado", 
        "lésbica", "transgênero", "LGBTQIA+", "vida sexual", "práticas sexuais",
        
        # Saúde
        "saúde", "prontuário médico", "doença", "enfermidade", "diagnóstico", 
        "tratamento médico", "medicamento", "condição de saúde", "deficiência", 
        "transtorno", "histórico médico", "exame", "resultado de exame", "internação", 
        "cirurgia", "HIV", "AIDS", "câncer", "diabetes", "hipertensão",
        
        # Dados genéticos e biométricos
        "dados genéticos", "DNA", "genoma", "código genético", "material genético", 
        "dados biométricos", "impressão digital", "reconhecimento facial", "íris", 
        "retina", "voz", "assinatura", "marcha",
        
        # Antecedentes criminais
        "antecedentes criminais", "processo criminal", "histórico judicial", 
        "condenação", "delito", "crime", "contravenção",
        
        # Outros dados sensíveis
        "biometria", "senhas", "geolocalização", "endereço IP", 
        "identificadores digitais"
    ]
    
    # Padrões para identificar informações pessoais
    padroes = {
        'CPF': r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',
        'CNPJ': r'\d{2}\.?\d{3}\.?\d{3}/?\.?\d{4}-?\d{2}',
        'RG': r'\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]',
        'TELEFONE': r'(\(?\d{2}\)?)\s*(\d{4,5})-?(\d{4})',
        'CEP': r'\d{5}-?\d{3}',
        'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'CARTAO_CREDITO': r'\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}',
        'DATA': r'\d{2}/\d{2}/\d{4}',
        'NOME_EMPRESA': r'([A-Z][A-ZÀ-Ú]+\s+){2,}([A-Z][A-ZÀ-Ú\s]*)+', # Sequência de palavras em maiúsculas
    }
    
    # Procurar e substituir os padrões
    for tipo, padrao in padroes.items():
        # Tratamento especial para nomes de empresas em maiúsculas
        if tipo == 'NOME_EMPRESA':
            # Usamos re.UNICODE para suportar caracteres acentuados
            for match in re.finditer(padrao, texto, re.UNICODE):
                info = match.group()
                # Verificar se tem pelo menos 3 palavras ou no mínimo 10 caracteres (para empresas com 2 palavras mais longas)
                palavras = [p for p in info.split() if p.strip()]
                if len(palavras) >= 2 and len(info) >= 10:
                    start = match.start()
                    end = match.end()
                    
                    # Armazenar o achado
                    achados.append({
                        'Tipo de Entidade': tipo,
                        'Texto': info,
                        'Início': start,
                        'Fim': end,
                        'Confiança': 0.95
                    })
                    
                    # Substituir a informação por asteriscos
                    mascara_texto = mascara * len(info)
                    texto = texto.replace(info, mascara_texto)
        else:
            for match in re.finditer(padrao, texto):
                info = match.group()
                start = match.start()
                end = match.end()
                
                # Armazenar o achado
                achados.append({
                    'Tipo de Entidade': tipo,
                    'Texto': info,
                    'Início': start,
                    'Fim': end,
                    'Confiança': 1.0
                })
                
                # Substituir a informação por asteriscos
                mascara_texto = mascara * len(info)
                texto = texto.replace(info, mascara_texto)
    
    # Procurar e substituir nomes comuns brasileiros
    nomes_comuns = [
        "Silva", "Santos", "Oliveira", "Souza", "Lima", "Pereira", "Ferreira", 
        "Almeida", "Costa", "Rodrigues", "Gomes", "Martins", "Araújo", "Carvalho",
        "João", "José", "Antonio", "Carlos", "Paulo", "Pedro", "Lucas", "Marcos", "Luis",
        "Gabriel", "Rafael", "Daniel", "Marcelo", "Bruno", "Eduardo", "Felipe", "Raimundo",
        "Maria", "Ana", "Francisca", "Antonia", "Adriana", "Juliana", "Marcia", "Fernanda",
        "Patricia", "Aline", "Sandra", "Camila", "Amanda", "Bruna", "Jessica", "Leticia"
    ]
    
    for nome in nomes_comuns:
        # Usando regex para encontrar nomes como palavras inteiras (não parte de outras palavras)
        padrao_nome = r'\b' + re.escape(nome) + r'\b'
        for match in re.finditer(padrao_nome, texto, re.IGNORECASE):
            info = match.group()
            start = match.start()
            end = match.end()
            
            # Armazenar o achado
            achados.append({
                'Tipo de Entidade': 'NOME',
                'Texto': info,
                'Início': start,
                'Fim': end,
                'Confiança': 0.85
            })
            
            # Substituir a informação por asteriscos
            mascara_texto = mascara * len(info)
            texto = texto.replace(info, mascara_texto)
    
    # Após os reconhecedores padrão, aplicar os termos sensíveis
    for termo in termos_sensiveis:
        padrao_termo = r'\b' + re.escape(termo) + r'\b'
        for match in re.finditer(padrao_termo, texto, re.IGNORECASE):
            info = match.group()
            start = match.start()
            end = match.end()
            
            # Armazenar o achado
            achados.append({
                'Tipo de Entidade': 'TERMO_SENSÍVEL',
                'Texto': info,
                'Início': start,
                'Fim': end,
                'Confiança': 0.90
            })
            
            # Substituir a informação por asteriscos
            mascara_texto = mascara * len(info)
            texto = texto.replace(info, mascara_texto)
    
    # Adicionar palavras personalizadas
    if palavras_adicionais:
        for palavra in palavras_adicionais:
            if palavra and len(palavra) > 2:  # Evitar palavras muito curtas
                padrao_palavra = r'\b' + re.escape(palavra) + r'\b'
                for match in re.finditer(padrao_palavra, texto, re.IGNORECASE):
                    info = match.group()
                    start = match.start()
                    end = match.end()
                    
                    # Armazenar o achado
                    achados.append({
                        'Tipo de Entidade': 'PERSONALIZADO',
                        'Texto': info,
                        'Início': start,
                        'Fim': end,
                        'Confiança': 0.95
                    })
                    
                    # Substituir a informação por asteriscos
                    mascara_texto = mascara * len(info)
                    texto = texto.replace(info, mascara_texto)
    
    return texto, achados

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF"""
    try:
        reader = PdfReader(pdf_file)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() + "\n"
        return texto
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return ""

def extract_text_from_csv(csv_file):
    """Extrai texto de um arquivo CSV"""
    try:
        df = pd.read_csv(csv_file)
        return df.to_string()
    except:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
            return df.to_string()
        except Exception as e:
            st.error(f"Erro ao ler CSV: {str(e)}")
            return ""

def process_file(uploaded_file, tolerancia, palavras, mascara):
    """Processa um arquivo enviado pelo usuário"""
    if uploaded_file.type == "application/pdf" or ".pdf" in uploaded_file.name.lower():
        texto = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/csv" or ".csv" in uploaded_file.name.lower():
        texto = extract_text_from_csv(uploaded_file)
    else:
        st.error("Tipo de arquivo não suportado")
        return None
    
    if not texto:
        st.error("Não foi possível extrair texto do arquivo.")
        return None
    
    return process_text(texto, tolerancia, palavras, mascara)

def process_text(texto, tolerancia, palavras, mascara):
    """Processa um texto para anonimização"""
    try:
        # Prepara a lista de palavras a serem negadas
        deny_list = [p.strip() for p in palavras.split(",")] if palavras else []
        
        # Usa a função de anonimização simples
        texto_anonimizado, achados = anonimizar_simples(texto, deny_list, mascara)
        
        # Se não encontrou nada para anonimizar
        if not achados:
            # Retorna o texto original com uma nota
            return {
                "texto": texto,
                "findings": [],
                "message": "Nenhuma informação pessoal foi identificada no texto."
            }
        
        # Prepara a resposta
        response = {
            "texto": texto_anonimizado,
            "findings": achados
        }
        
        return response
        
    except Exception as e:
        st.error(f"Erro ao anonimizar texto: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.title("Anonimizador de Textos - LGPD")
    st.markdown("""
    ### Proteja dados sensíveis de acordo com a Lei Geral de Proteção de Dados
    Esta ferramenta ajuda a identificar e mascarar informações pessoais identificáveis (PII) em textos.
    """)
    
    # Opções de configuração
    with st.expander("Configurações de Anonimização", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            tolerancia = st.slider(
                "Tolerância de detecção", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.25,
                help="Valores menores detectam mais padrões, mas podem gerar falsos positivos"
            )
        with col2:
            palavras = st.text_input(
                "Palavras adicionais a mascarar", 
                placeholder="Palavras separadas por vírgula (sem espaços)",
                help="Lista de palavras específicas que devem ser detectadas"
            )
        with col3:
            mascara = st.text_input(
                "Caractere de máscara",
                value="*",
                max_chars=1,
                help="Caractere usado para substituir informações pessoais"
            )
    
    # Criação de abas para os diferentes métodos de input
    tab1, tab2 = st.tabs(["Texto", "Arquivo (PDF/CSV)"])
    
    with tab1:
        st.subheader("Anonimização de Texto")
        user_input = st.text_area(
            "Digite ou cole o texto:", 
            height=200,
            placeholder="Cole aqui o texto que deseja anonimizar...",
        )
        
        if st.button("Anonimizar Texto", type="primary", key="btn_texto"):
            if user_input:
                with st.spinner("Processando texto..."):
                    resultado = process_text(user_input, tolerancia, palavras, mascara)
                    if resultado:
                        if "message" in resultado:
                            st.info(resultado["message"])
                        else:
                            st.success("Texto anonimizado com sucesso!")
                        
                        # Display the findings
                        if "findings" in resultado and resultado["findings"]:
                            with st.expander("Informações detectadas", expanded=True):
                                findings_df = pd.DataFrame(resultado["findings"])
                                st.dataframe(findings_df)
                        
                        # Display the anonymized text
                        st.subheader("Resultado:")
                        st.markdown(
                            f"""<div style="padding: 15px; border-radius: 5px; background-color: #f0f2f6;">
                            {resultado["texto"]}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        
                        # Download option
                        text_download = resultado["texto"]
                        st.download_button(
                            label="Baixar texto anonimizado",
                            data=text_download,
                            file_name="texto_anonimizado.txt",
                            mime="text/plain"
                        )
            else:
                st.warning("Por favor, insira um texto para anonimizar.")
    
    with tab2:
        st.subheader("Anonimização de Arquivo")
        uploaded_file = st.file_uploader("Faça upload do arquivo:", type=["pdf", "csv"])
        
        if uploaded_file is not None:
            file_details = {"Nome": uploaded_file.name, "Tipo": uploaded_file.type, "Tamanho": f"{uploaded_file.size/1024:.2f} KB"}
            st.write(file_details)
            
            if st.button("Anonimizar Arquivo", type="primary", key="btn_arquivo"):
                with st.spinner(f"Processando arquivo {uploaded_file.name}..."):
                    resultado = process_file(uploaded_file, tolerancia, palavras, mascara)
                    if resultado:
                        if "message" in resultado:
                            st.info(resultado["message"])
                        else:
                            st.success("Arquivo anonimizado com sucesso!")
                        
                        # Display the findings
                        if "findings" in resultado and resultado["findings"]:
                            with st.expander("Informações detectadas", expanded=True):
                                findings_df = pd.DataFrame(resultado["findings"])
                                st.dataframe(findings_df)
                        
                        # Display the anonymized text
                        st.subheader("Resultado:")
                        st.text_area("Texto anonimizado", value=resultado["texto"], height=400)
                        
                        # Download option
                        text_download = resultado["texto"]
                        st.download_button(
                            label="Baixar texto anonimizado",
                            data=text_download,
                            file_name=f"{uploaded_file.name}_anonimizado.txt",
                            mime="text/plain"
                        )

    # Informações adicionais
    with st.expander("Sobre este anonimizador"):
        st.markdown("""
        ### Sobre o Anonimizador de Textos
        
        **Objetivos:**
        - Fornecer uma forma simples de preservar a privacidade de dados pessoais
        - Permitir customização para atender a necessidades específicas
        - Facilitar a detecção automática e semi-automática de Informações Pessoais Identificáveis (PII)
        
        **Limitações:**
        > ⚠️ **Atenção:** O anonimizador pode ajudar a identificar dados sensíveis em textos, mas por ser um 
        > mecanismo de detecção automática, não há garantias de que todas as informações sensíveis serão encontradas.
        > Sistemas adicionais de proteção devem ser empregados.
        
        **Tipos de dados detectados:**
        - CPF/CNPJ
        - RG
        - Nomes comuns brasileiros
        - Nomes de empresas e organizações (sequências de palavras em maiúsculas)
        - Endereços e CEPs
        - Números de telefone
        - E-mails
        - Dados de cartão de crédito
        - Datas
        - Termos sensíveis (raça, religião, orientação sexual, saúde, dados biométricos, etc.)
        - Palavras personalizadas definidas pelo usuário
        """)
    
    # Rodapé
    st.markdown("---")
    st.markdown("© 2025 Anonimizador de Textos - LGPD | Desenvolvido com Streamlit")

if __name__ == "__main__":
    main()
