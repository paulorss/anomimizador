import streamlit as st
import base64
import io
import re
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader
import regex as re
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configuração da página
st.set_page_config(
    page_title="Anonimizador de Textos - LGPD",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="auto",
)

def create_pt_br_recognizers():
    """
    Cria reconhecedores personalizados para o português brasileiro
    """
    recognizers = []
    
    # CPF Recognizer
    cpf_pattern = Pattern(
        name="cpf_pattern",
        regex=r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',
        score=0.9
    )
    cpf_recognizer = PatternRecognizer(
        supported_entity="CPF",
        patterns=[cpf_pattern],
        context=["cpf", "cadastro de pessoa física", "documento", "número"]
    )
    recognizers.append(cpf_recognizer)
    
    # CNPJ Recognizer
    cnpj_pattern = Pattern(
        name="cnpj_pattern",
        regex=r'\d{2}\.?\d{3}\.?\d{3}/?\.?\d{4}-?\d{2}',
        score=0.9
    )
    cnpj_recognizer = PatternRecognizer(
        supported_entity="CNPJ",
        patterns=[cnpj_pattern],
        context=["cnpj", "cadastro nacional", "empresa", "pessoa jurídica"]
    )
    recognizers.append(cnpj_recognizer)
    
    # RG Recognizer
    rg_pattern = Pattern(
        name="rg_pattern",
        regex=r'\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]',
        score=0.7
    )
    rg_recognizer = PatternRecognizer(
        supported_entity="RG", 
        patterns=[rg_pattern],
        context=["rg", "registro geral", "identidade", "documento"]
    )
    recognizers.append(rg_recognizer)
    
    # Telefone Brasileiro
    telefone_pattern = Pattern(
        name="telefone_pattern",
        regex=r'(\(?\d{2}\)?)\s*(\d{4,5})-?(\d{4})|\(\d{2}\)\s*\d{4,5}-?\d{4}|(\d{2})\s*9?\d{4}-?\d{4}',
        score=0.8
    )
    telefone_recognizer = PatternRecognizer(
        supported_entity="TELEFONE",
        patterns=[telefone_pattern],
        context=["telefone", "celular", "contato", "ligar", "whatsapp"]
    )
    recognizers.append(telefone_recognizer)
    
    # CEP Brasileiro
    cep_pattern = Pattern(
        name="cep_pattern",
        regex=r'\d{5}-?\d{3}',
        score=0.8
    )
    cep_recognizer = PatternRecognizer(
        supported_entity="CEP",
        patterns=[cep_pattern],
        context=["cep", "código postal", "endereço"]
    )
    recognizers.append(cep_recognizer)
    
    # Endereço
    endereco_pattern = Pattern(
        name="endereco_pattern",
        regex=r'(rua|avenida|av\.|alameda|praça|travessa|rod\.|rodovia)\s+[A-Za-zÀ-ÿ\s\.\,0-9]+,?\s*(n°\.?|nº\.?|número\.?)?\s*\d*',
        score=0.65
    )
    endereco_recognizer = PatternRecognizer(
        supported_entity="ENDERECO",
        patterns=[endereco_pattern],
        context=["endereço", "localizado", "reside", "mora"]
    )
    recognizers.append(endereco_recognizer)
    
    # Cartão de Crédito
    cartao_pattern = Pattern(
        name="cartao_pattern",
        regex=r'\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}',
        score=0.9
    )
    cartao_recognizer = PatternRecognizer(
        supported_entity="CARTAO_CREDITO",
        patterns=[cartao_pattern],
        context=["cartão", "crédito", "débito", "visa", "master"]
    )
    recognizers.append(cartao_recognizer)
    
    # Data de Nascimento
    data_pattern = Pattern(
        name="data_pattern",
        regex=r'nascido\s+em\s+\d{2}[./]\d{2}[./]\d{4}|nascida\s+em\s+\d{2}[./]\d{2}[./]\d{4}|data\s+de\s+nascimento:?\s+\d{2}[./]\d{2}[./]\d{4}',
        score=0.8
    )
    data_recognizer = PatternRecognizer(
        supported_entity="DATA_NASCIMENTO",
        patterns=[data_pattern],
        context=["nascimento", "nascido", "nascida", "idade"]
    )
    recognizers.append(data_recognizer)
    
    # Filiação
    filiacao_pattern = Pattern(
        name="filiacao_pattern",
        regex=r'filho\s+(de|da|do)\s+[A-Za-zÀ-ÿ\s]+(\s+e\s+de\s+[A-Za-zÀ-ÿ\s]+)?',
        score=0.7
    )
    filiacao_recognizer = PatternRecognizer(
        supported_entity="FILIACAO",
        patterns=[filiacao_pattern],
        context=["filho", "filha", "pai", "mãe", "genitores"]
    )
    recognizers.append(filiacao_recognizer)
    
    # Estado Civil
    estado_civil_pattern = Pattern(
        name="estado_civil_pattern",
        regex=r'estado\s+civil\s*:?\s*(casado|solteiro|viúvo|divorciado|separado|união estável)',
        score=0.7
    )
    estado_civil_recognizer = PatternRecognizer(
        supported_entity="ESTADO_CIVIL",
        patterns=[estado_civil_pattern],
        context=["estado civil", "casado", "solteiro", "viúvo", "divorciado"]
    )
    recognizers.append(estado_civil_recognizer)
    
    # Idade
    idade_pattern = Pattern(
        name="idade_pattern",
        regex=r'\b\d{1,3}\s+anos\b|\bidade\s+de\s+\d{1,3}\b',
        score=0.6
    )
    idade_recognizer = PatternRecognizer(
        supported_entity="IDADE",
        patterns=[idade_pattern],
        context=["idade", "anos", "aniversário"]
    )
    recognizers.append(idade_recognizer)
    
    # Profissão
    profissao_pattern = Pattern(
        name="profissao_pattern",
        regex=r'profissão\s*:?\s*[A-Za-zÀ-ÿ\s]+',
        score=0.6
    )
    profissao_recognizer = PatternRecognizer(
        supported_entity="PROFISSAO",
        patterns=[profissao_pattern],
        context=["trabalha", "emprego", "ocupação", "cargo"]
    )
    recognizers.append(profissao_recognizer)
    
    return recognizers

def create_custom_deny_list_recognizer(deny_list):
    """
    Cria um reconhecedor baseado em uma lista personalizada de palavras
    """
    if not deny_list or not any(deny_list):
        return None
        
    patterns = []
    for i, word in enumerate(deny_list):
        if word and len(word) > 2:  # Ignora palavras muito curtas
            word_pattern = Pattern(
                name=f"custom_pattern_{i}",
                regex=r'\b' + re.escape(word) + r'\b',
                score=0.7
            )
            patterns.append(word_pattern)
            
    if patterns:
        return PatternRecognizer(
            supported_entity="CUSTOM",
            patterns=patterns,
            context=[]
        )
    return None

def setup_analyzer_engine(deny_list=None):
    """
    Configura o motor de análise do Presidio com os reconhecedores personalizados
    """
    # Criar um registro de reconhecedores
    registry = RecognizerRegistry(supported_languages=["pt"])
    
    # Configurar o provedor de NLP
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "pt", "model_name": "pt_core_news_md"}]
    }).create_engine()
    
    # Adicionar os reconhecedores padrão da Presidio com suporte a PT
    registry.load_predefined_recognizers(languages=["pt"])
    
    # Adicionar os reconhecedores personalizados
    for recognizer in create_pt_br_recognizers():
        registry.add_recognizer(recognizer)
    
    # Adicionar o reconhecedor personalizado de lista negada
    if deny_list:
        custom_deny_list_recognizer = create_custom_deny_list_recognizer(deny_list)
        if custom_deny_list_recognizer:
            registry.add_recognizer(custom_deny_list_recognizer)
    
    # Criar o motor de análise
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["pt"]
    )
    
    return analyzer

def setup_anonymizer_engine():
    """
    Configura o motor de anonimização do Presidio
    """
    return AnonymizerEngine()

def anonimizar_com_presidio(texto, deny_list=None, mascara='*'):
    """
    Anonimiza o texto usando o Presidio
    """
    if not texto:
        return texto, []
    
    # Configurar o motor de análise
    analyzer = setup_analyzer_engine(deny_list)
    
    # Configurar o motor de anonimização
    anonymizer = setup_anonymizer_engine()
    
    # Analisar o texto
    results = analyzer.analyze(
        text=texto,
        language="pt",
        entities=None,  # Detectar todas as entidades suportadas
        allow_list=None,
        score_threshold=0.4  # Ajustar conforme necessário
    )
    
    # Configurar operações de anonimização
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": mascara * 5}),
        "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": 10, "from_end": False}),
        "CREDIT_CARD": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": 12, "from_end": True}),
        "CUSTOM": OperatorConfig("replace", {"new_value": mascara * 7}),
    }
    
    # Converter os resultados do Presidio para o formato esperado pelo código original
    achados = []
    for result in results:
        achados.append({
            'Tipo de Entidade': result.entity_type,
            'Texto': texto[result.start:result.end],
            'Início': result.start,
            'Fim': result.end,
            'Confiança': result.score
        })
    
    # Anonimizar o texto se houver achados
    if results:
        try:
            anonymized_result = anonymizer.anonymize(
                text=texto,
                analyzer_results=results,
                operators=operators
            )
            texto_anonimizado = anonymized_result.text
        except Exception as e:
            st.error(f"Erro na anonimização: {str(e)}")
            # Fallback para método mais simples se o anonymizer falhar
            texto_anonimizado = texto
            for result in results:
                texto_anonimizado = texto_anonimizado.replace(
                    texto[result.start:result.end],
                    mascara * len(texto[result.start:result.end])
                )
    else:
        texto_anonimizado = texto
    
    return texto_anonimizado, achados

def anonimizar_hibrido(texto, palavras_adicionais=None, mascara='*', tolerancia=0.5):
    """
    Função híbrida que combina o Presidio com o método simples para melhor cobertura
    """
    if not texto:
        return texto, []
    
    # Lista para armazenar palavras adicionais
    deny_list = [p.strip() for p in palavras_adicionais.split(",")] if palavras_adicionais else []
    
    # Primeiro, usar Presidio para anonimização
    texto_presidio, achados_presidio = anonimizar_com_presidio(texto, deny_list, mascara)
    
    # Se o Presidio não encontrou nada ou encontrou poucos dados, tentar o método simples como backup
    if len(achados_presidio) < 3:  # Número arbitrário para decidir se vale a pena tentar o método simples
        texto_simples, achados_simples = anonimizar_simples(texto, deny_list, mascara)
        
        # Comparar resultados e escolher o que encontrou mais achados
        if len(achados_simples) > len(achados_presidio):
            return texto_simples, achados_simples
    
    return texto_presidio, achados_presidio

# Função anonimizar_simples mantida para backup
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
        "orientação sexual", "heterossexual", "homossexual", "bissexual", "gay", 
        "lésbica", "transgênero", "LGBTQIA+", "vida sexual", "práticas sexuais",
        
        # Estado civil
        "casado", "solteiro", "viúvo", "viúva", "casada", "divorciado", "divorciada",
        "separado", "separada", "união estável",
        
        # Profissão
        "advogado", "médico", "médica", "engenheiro", "engenheira", "professor", "professora",
        "contador", "contadora", "dentista", "enfermeiro", "enfermeira", "arquiteto", "arquiteta",
        "policial", "motorista", "vendedor", "vendedora", "empresário", "empresária", "autônomo",
        "autônoma", "desempregado", "desempregada", "aposentado", "aposentada", "estudante",
        "estagiário", "estagiária", "funcionário público", "funcionária pública",
        
        # Filiação
        "filho de", "filha de", "pai", "mãe", "filiação", "filho", "filha", "genitor", "genitora",
        
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
        'TELEFONE': r'(\(?\d{2}\)?)\s*(\d{4,5})-?(\d{4})|\(\d{2}\)\s*\d{4,5}-?\d{4}|(\d{2})\s*9?\d{4}-?\d{4}',
        'CEP': r'\d{5}-?\d{3}',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|[Ee]-?mail:?\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        'CARTAO_CREDITO': r'\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}',
        'DATA': r'\d{2}[./]\d{2}[./]\d{4}|\d{2}-\d{2}-\d{4}',
        'NOME_EMPRESA': r'([A-Z][A-ZÀ-Ú]+\s+){2,}([A-Z][A-ZÀ-Ú\s]*)+', # Sequência de palavras em maiúsculas
        'IDADE': r'\b\d{1,3}\s+anos\b|\bidade\s+de\s+\d{1,3}\b',
        'DATA_NASCIMENTO': r'nascido\s+em\s+\d{2}[./]\d{2}[./]\d{4}|nascida\s+em\s+\d{2}[./]\d{2}[./]\d{4}|data\s+de\s+nascimento:?\s+\d{2}[./]\d{2}[./]\d{4}',
        'ENDERECO': r'residente\s+(e\s+domiciliado\s+)?(a|à|na|no)?\s+.{5,50}(,\s+n°\.?\s+\d+)?|morador\s+(a|à|na|no)?\s+.{5,50}(,\s+n°\.?\s+\d+)?',
        'ENDERECO_RUA': r'(rua|avenida|av\.|alameda|praça|travessa|rod\.|rodovia)\s+[A-Za-zÀ-ÿ\s\.\,0-9]+,?\s*(n°\.?|nº\.?|número\.?)?\s*\d*',
        'ENDERECO_BAIRRO': r'(bairro|b\.)\s+[A-Za-zÀ-ÿ\s]+',
        'ENDERECO_CIDADE': r'(cidade|cid\.|município) de\s+[A-Za-zÀ-ÿ\s]+|[Ee]m\s+[A-Z][a-zÀ-ÿ]+(/[A-Z]{2})?',
        'ESTADO_CIVIL': r'estado\s+civil\s*:?\s*(casado|solteiro|viúvo|divorciado|separado|união estável)',
        'PROFISSAO': r'profissão\s*:?\s*[A-Za-zÀ-ÿ\s]+',
        'FILIACAO': r'filho\s+(de|da|do)\s+[A-Za-zÀ-ÿ\s]+(\s+e\s+de\s+[A-Za-zÀ-ÿ\s]+)?',
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
        # Tratamento especial para endereços e outros padrões longos
        elif tipo in ['ENDERECO', 'ENDERECO_RUA', 'PROFISSAO', 'FILIACAO']:
            for match in re.finditer(padrao, texto, re.IGNORECASE):
                info = match.group()
                start = match.start()
                end = match.end()
                
                # Armazenar o achado
                achados.append({
                    'Tipo de Entidade': tipo,
                    'Texto': info,
                    'Início': start,
                    'Fim': end,
                    'Confiança': 0.9
                })
                
                # Substituir a informação por asteriscos
                mascara_texto = mascara * len(info)
                texto = texto.replace(info, mascara_texto)
        else:
            for match in re.finditer(padrao, texto, re.IGNORECASE):
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
    """Processa um texto para anonimização, usando uma abordagem híbrida (Presidio + Regex)"""
    try:
        # Verifica o modo de anonimização selecionado
        modo = st.session_state.get("modo_anonimizacao", "Automático (recomendado)")
        
        # Prepara a lista de palavras a serem negadas
        deny_list = [p.strip() for p in palavras.split(",")] if palavras else []
        
        # Escolhe a função de anonimização com base no modo selecionado
        if modo == "Presidio":
            texto_anonimizado, achados = anonimizar_com_presidio(texto, deny_list, mascara)
        elif modo == "Regex":
            texto_anonimizado, achados = anonimizar_simples(texto, deny_list, mascara)
        else:  # Automático
            texto_anonimizado, achados = anonimizar_hibrido(texto, palavras, mascara, tolerancia)
        
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
    Esta ferramenta ajuda a identificar e mascarar informações pessoais identificáveis (PII) em textos, 
    utilizando a biblioteca Microsoft Presidio e técnicas avançadas de reconhecimento de padrões para o português brasileiro.
    """)
    
    # Opções de configuração
    with st.expander("Configurações de Anonimização", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            tolerancia = st.slider(
                "Tolerância de detecção", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.4,
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
            
        # Modo de anonimização
        st.radio(
            "Modo de anonimização",
            ["Automático (recomendado)", "Presidio", "Regex"],
            index=0,
            key="modo_anonimizacao",
            help="Escolha o método de detecção de informações pessoais. O modo Automático combina ambas as técnicas."
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
                                # Ordenar por confiança (decrescente)
                                findings_df = findings_df.sort_values(by='Confiança', ascending=False)
                                st.dataframe(findings_df)
                                
                                # Métricas de resumo
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total de itens identificados", len(findings_df))
                                with col2:
                                    num_tipos = findings_df['Tipo de Entidade'].nunique()
                                    st.metric("Tipos de dados diferentes", num_tipos)
                                with col3:
                                    confianca_media = findings_df['Confiança'].mean()
                                    st.metric("Confiança média", f"{confianca_media:.2f}")
                        
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
                        
                        # Opção para relatório
                        if "findings" in resultado and resultado["findings"]:
                            report = generate_anonymization_report(user_input, resultado)
                            st.download_button(
                                label="Baixar relatório detalhado",
                                data=report,
                                file_name="relatorio_anonimizacao.txt",
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
                                # Ordenar por confiança (decrescente)
                                findings_df = findings_df.sort_values(by='Confiança', ascending=False)
                                st.dataframe(findings_df)
                                
                                # Métricas de resumo
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total de itens identificados", len(findings_df))
                                with col2:
                                    num_tipos = findings_df['Tipo de Entidade'].nunique()
                                    st.metric("Tipos de dados diferentes", num_tipos)
                                with col3:
                                    confianca_media = findings_df['Confiança'].mean()
                                    st.metric("Confiança média", f"{confianca_media:.2f}")
                        
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
                        
                        # Opção para relatório
                        if "findings" in resultado and resultado["findings"]:
                            report = generate_anonymization_report(
                                f"Arquivo: {uploaded_file.name}", 
                                resultado
                            )
                            st.download_button(
                                label="Baixar relatório detalhado",
                                data=report,
                                file_name=f"{uploaded_file.name}_relatorio.txt",
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
        - Atender às exigências da LGPD (Lei Geral de Proteção de Dados Pessoais)
        
        **Tecnologias utilizadas:**
        - Microsoft Presidio: Framework de código aberto para anonimização de dados
        - Reconhecimento de Entidades Nomeadas (NER)
        - Expressões regulares otimizadas para o português brasileiro
        - Abordagem híbrida para maximizar a detecção de dados pessoais
        
        **Limitações:**
        > ⚠️ **Atenção:** O anonimizador pode ajudar a identificar dados sensíveis em textos, mas por ser um 
        > mecanismo de detecção automática, não há garantias de que todas as informações sensíveis serão encontradas.
        > Sistemas adicionais de proteção devem ser empregados.
        > Sempre revise o texto anonimizado manualmente antes de usá-lo em produção.
        
        **Tipos de dados detectados:**
        - CPF/CNPJ
        - RG
        - Endereço completo e partes (rua, bairro, cidade)
        - Estado civil
        - Profissão
        - Filiação
        - Data de nascimento
        - Idade
        - Nomes comuns brasileiros
        - Nomes de empresas e organizações
        - CEPs
        - Números de telefone
        - E-mails
        - Dados de cartão de crédito
        - Datas
        - Termos sensíveis (raça, religião, orientação sexual, saúde, dados biométricos, etc.)
        - Palavras personalizadas definidas pelo usuário
        
        **Conformidade com a LGPD:**
        Esta ferramenta ajuda a implementar medidas técnicas para proteção de dados pessoais,
        conforme requerido pelos artigos 46-49 da LGPD (Lei 13.709/2018), contribuindo para
        a minimização de riscos e o tratamento adequado de dados sensíveis.
        """)
    
    # Rodapé
    st.markdown("---")
    st.markdown("© 2025 Anonimizador de Textos - LGPD | Desenvolvido com Streamlit e Microsoft Presidio")

def generate_anonymization_report(original_text, resultado):
    """Gera um relatório detalhado da anonimização"""
    findings = resultado.get("findings", [])
    texto_anonimizado = resultado.get("texto", "")
    
    report = "RELATÓRIO DE ANONIMIZAÇÃO DE DADOS\n"
    report += "=" * 50 + "\n\n"
    
    # Data e hora
    from datetime import datetime
    report += f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
    
    # Resumo
    report += "RESUMO:\n"
    report += "-" * 50 + "\n"
    report += f"Total de informações sensíveis encontradas: {len(findings)}\n"
    
    if findings:
        # Contar tipos de entidades
        tipos_entidade = {}
        for finding in findings:
            tipo = finding.get('Tipo de Entidade', 'Desconhecido')
            tipos_entidade[tipo] = tipos_entidade.get(tipo, 0) + 1
        
        report += "Distribuição por tipo:\n"
        for tipo, contagem in tipos_entidade.items():
            report += f"  - {tipo}: {contagem}\n"
        
        # Confiança média
        confianca_total = sum(finding.get('Confiança', 0) for finding in findings)
        confianca_media = confianca_total / len(findings)
        report += f"Confiança média: {confianca_media:.2f}\n\n"
        
        # Detalhes das informações encontradas
        report += "DETALHES DAS INFORMAÇÕES ENCONTRADAS:\n"
        report += "-" * 50 + "\n"
        
        for i, finding in enumerate(findings, 1):
            report += f"Item {i}:\n"
            report += f"  Tipo: {finding.get('Tipo de Entidade', 'Desconhecido')}\n"
            report += f"  Texto: {finding.get('Texto', '')}\n"
            report += f"  Confiança: {finding.get('Confiança', 0):.2f}\n"
            report += f"  Posição: {finding.get('Início', 0)}-{finding.get('Fim', 0)}\n"
            report += "\n"
    
    # Amostras de texto (original e anonimizado)
    # Limitando para evitar relatórios muito grandes
    max_sample_length = 1000
    sample_original = original_text[:max_sample_length] + ("..." if len(original_text) > max_sample_length else "")
    sample_anon = texto_anonimizado[:max_sample_length] + ("..." if len(texto_anonimizado) > max_sample_length else "")
    
    report += "AMOSTRA DE TEXTO ORIGINAL:\n"
    report += "-" * 50 + "\n"
    report += sample_original + "\n\n"
    
    report += "AMOSTRA DE TEXTO ANONIMIZADO:\n"
    report += "-" * 50 + "\n"
    report += sample_anon + "\n\n"
    
    # Recomendações
    report += "RECOMENDAÇÕES:\n"
    report += "-" * 50 + "\n"
    report += "1. Verifique manualmente o texto anonimizado para garantir que todas as informações sensíveis foram detectadas.\n"
    report += "2. Considere ajustar a tolerância de detecção se informações importantes não foram identificadas.\n"
    report += "3. Para casos específicos, adicione palavras personalizadas à lista de detecção.\n"
    report += "4. Lembre-se que esta é uma ferramenta de auxílio e não substitui uma revisão manual cuidadosa.\n\n"
    
    report += "=" * 50 + "\n"
    report += "Este relatório foi gerado automaticamente pelo Anonimizador de Textos - LGPD.\n"
    
    return report

if __name__ == "__main__":
    main()
