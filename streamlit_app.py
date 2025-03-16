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
import spacy
import traceback
import time
from datetime import datetime

# Verificar e carregar modelo spaCy se não estiver disponível
try:
    nlp = spacy.load("pt_core_news_md")
except OSError:
    import sys
    try:
        st.info("Baixando modelo de linguagem para análise de texto. Isso pode levar alguns minutos...")
        spacy.cli.download("pt_core_news_md")
        nlp = spacy.load("pt_core_news_md")
    except:
        st.error("Não foi possível baixar o modelo de linguagem. Usando método de fallback.")
        nlp = None

# Configuração da página
st.set_page_config(
    page_title="Anonimizador de Textos - LGPD",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="auto",
)

# Inicializar variáveis de estado da sessão
if 'modo_anonimizacao' not in st.session_state:
    st.session_state.modo_anonimizacao = "Automático (recomendado)"


def create_pt_br_recognizers():
    """
    Cria reconhecedores personalizados para o português brasileiro com padrões melhorados
    """
    recognizers = []
    
    # CPF Recognizer - Melhorado para detectar formatos com e sem pontuação
    cpf_pattern = Pattern(
        name="cpf_pattern",
        regex=r'\b\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\.\s]?\d{2}\b',
        score=0.95
    )
    cpf_recognizer = PatternRecognizer(
        supported_entity="CPF",
        patterns=[cpf_pattern],
        context=["cpf", "cadastro", "pessoa física", "documento", "número", "contribuinte"]
    )
    recognizers.append(cpf_recognizer)
    
    # CNPJ Recognizer - Melhorado para detectar formatos com e sem pontuação
    cnpj_pattern = Pattern(
        name="cnpj_pattern",
        regex=r'\b\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[/\.\s]?\d{4}[-\.\s]?\d{2}\b',
        score=0.95
    )
    cnpj_recognizer = PatternRecognizer(
        supported_entity="CNPJ",
        patterns=[cnpj_pattern],
        context=["cnpj", "cadastro nacional", "empresa", "pessoa jurídica", "inscrição"]
    )
    recognizers.append(cnpj_recognizer)
    
    # RG Recognizer - Melhorado para cobrir mais formatos de RG usados no Brasil
    rg_pattern = Pattern(
        name="rg_pattern",
        regex=r'\b\d{1,2}[\.\s]?\d{3}[\.\s]?\d{3}[-\.\s]?[\dxX]\b|\b[A-Za-z]{2}[-\.\s]?\d{6,8}\b',
        score=0.8
    )
    rg_recognizer = PatternRecognizer(
        supported_entity="RG", 
        patterns=[rg_pattern],
        context=["rg", "registro geral", "identidade", "documento", "cédula", "identificação"]
    )
    recognizers.append(rg_recognizer)
    
    # Telefone Brasileiro - Padrão melhorado para cobrir mais formatos
    telefone_pattern = Pattern(
        name="telefone_pattern",
        regex=r'\(\s*\d{2}\s*\)\s*9?\s*\d{4}[-\.\s]?\d{4}|\b\d{2}\s*9?\s*\d{4}[-\.\s]?\d{4}\b|\b\d{5}[-\.\s]?\d{4}\b',
        score=0.85
    )
    telefone_recognizer = PatternRecognizer(
        supported_entity="TELEFONE",
        patterns=[telefone_pattern],
        context=["telefone", "celular", "contato", "ligar", "whatsapp", "fone", "tel", "tel:", "telefone:", "celular:"]
    )
    recognizers.append(telefone_recognizer)
    
    # CEP Brasileiro - Padrão melhorado
    cep_pattern = Pattern(
        name="cep_pattern",
        regex=r'\b\d{5}[-\.\s]?\d{3}\b',
        score=0.9
    )
    cep_recognizer = PatternRecognizer(
        supported_entity="CEP",
        patterns=[cep_pattern],
        context=["cep", "código postal", "endereço", "código de endereçamento", "cep:"]
    )
    recognizers.append(cep_recognizer)
    
    # Email - Melhorado para detectar emails em vários contextos
    email_pattern = Pattern(
        name="email_pattern",
        regex=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|[Ee]-?mail:?\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        score=0.95
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL",
        patterns=[email_pattern],
        context=["email", "correio eletrônico", "e-mail", "@", "contato"]
    )
    recognizers.append(email_recognizer)

# Endereço - Padrão melhorado para pegar ruas, avenidas, etc.
    endereco_pattern = Pattern(
        name="endereco_pattern",
        regex=r'\b(rua|avenida|av\.|alameda|praça|travessa|rod\.|rodovia|estrada|quadra|lote|bloco|setor|via)\s+[A-Za-zÀ-ÿ\s\.\,0-9]+(,?\s*n[º°\.]\s*\d+)?(\s*[-,]\s*[A-Za-zÀ-ÿ\s\.\,0-9]+)?',
        score=0.7
    )
    endereco_recognizer = PatternRecognizer(
        supported_entity="ENDERECO",
        patterns=[endereco_pattern],
        context=["endereço", "localizado", "reside", "mora", "domiciliado", "residente", "localidade", "endereço:"]
    )
    recognizers.append(endereco_recognizer)
    
    # Bairro
    bairro_pattern = Pattern(
        name="bairro_pattern",
        regex=r'\b(bairro|b\.)\s+[A-Za-zÀ-ÿ\s]+',
        score=0.7
    )
    bairro_recognizer = PatternRecognizer(
        supported_entity="BAIRRO",
        patterns=[bairro_pattern],
        context=["bairro", "comunidade", "vila", "jardim", "residencial", "bairro:"]
    )
    recognizers.append(bairro_recognizer)
    
    # Cartão de Crédito - Padrão melhorado
    cartao_pattern = Pattern(
        name="cartao_pattern",
        regex=r'\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b',
        score=0.95
    )
    cartao_recognizer = PatternRecognizer(
        supported_entity="CARTAO_CREDITO",
        patterns=[cartao_pattern],
        context=["cartão", "crédito", "débito", "visa", "master", "elo", "american", "nubank", "itaucard", "hipercard"]
    )
    recognizers.append(cartao_recognizer)
    
    # Data de Nascimento - Padrão melhorado para detectar mais formatos
    data_pattern = Pattern(
        name="data_pattern",
        regex=r'nascido\s+em\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}|nascida\s+em\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}|data\s+de\s+nascimento:?\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}|\bnasc\.?\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}',
        score=0.85
    )
    data_recognizer = PatternRecognizer(
        supported_entity="DATA_NASCIMENTO",
        patterns=[data_pattern],
        context=["nascimento", "nascido", "nascida", "idade", "aniversário", "natalício"]
    )
    recognizers.append(data_recognizer)
    
    # Filiação - Padrão melhorado
    filiacao_pattern = Pattern(
        name="filiacao_pattern",
        regex=r'filho\s+(de|da|do)\s+[A-Za-zÀ-ÿ\s]+((\s+e\s+)|(\s*,\s*))?([A-Za-zÀ-ÿ\s]+)?',
        score=0.8
    )
    filiacao_recognizer = PatternRecognizer(
        supported_entity="FILIACAO",
        patterns=[filiacao_pattern],
        context=["filho", "filha", "pai", "mãe", "genitores", "pais", "filiação"]
    )
    recognizers.append(filiacao_recognizer)

# Nome Completo - Usando regras para nomes
    nome_pattern = Pattern(
        name="nome_pattern",
        regex=r'\b([A-Z][a-zÀ-ÿ]+\s+){1,2}([A-Z][a-zÀ-ÿ]+\s+)?([A-Z][a-zÀ-ÿ]+(\s+[A-Z][a-zÀ-ÿ]+)?){1,3}\b',
        score=0.7
    )
    nome_recognizer = PatternRecognizer(
        supported_entity="NOME_COMPLETO",
        patterns=[nome_pattern],
        context=["nome", "chamado", "conhecido", "sr.", "sra.", "srta.", "doutor", "dr.", "paciente"]
    )
    recognizers.append(nome_recognizer)
    
    # PIS/PASEP
    pis_pasep_pattern = Pattern(
        name="pis_pasep_pattern",
        regex=r'\b\d{3}\.?\d{5}\.?\d{2}-?\d{1}\b',
        score=0.9
    )
    pis_pasep_recognizer = PatternRecognizer(
        supported_entity="PIS_PASEP",
        patterns=[pis_pasep_pattern],
        context=["pis", "pasep", "nis", "inss", "previdência", "contribuição"]
    )
    recognizers.append(pis_pasep_recognizer)
    
    # Título de Eleitor
    titulo_eleitor_pattern = Pattern(
        name="titulo_eleitor_pattern",
        regex=r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        score=0.8
    )
    titulo_eleitor_recognizer = PatternRecognizer(
        supported_entity="TITULO_ELEITOR",
        patterns=[titulo_eleitor_pattern],
        context=["título", "eleitor", "eleitoral", "votação", "zona", "seção"]
    )
    recognizers.append(titulo_eleitor_recognizer)
    
    # CNH
    cnh_pattern = Pattern(
        name="cnh_pattern",
        regex=r'\bcnh\s*:?\s*\d{9,11}\b|\bcarteira\s+de\s+habilitação\s*:?\s*\d{9,11}\b',
        score=0.7
    )
    cnh_recognizer = PatternRecognizer(
        supported_entity="CNH",
        patterns=[cnh_pattern],
        context=["cnh", "carteira", "habilitação", "motorista", "direção", "condutor"]
    )
    recognizers.append(cnh_recognizer)
    
    # Dados Bancários
    banco_pattern = Pattern(
        name="banco_pattern",
        regex=r'(banco|agência|ag[\.:]?|conta[\.:]?)\s*\d+[-\.\s]?\d*|agência\s*\d+[-\.\s]?\d*\s*,?\s*conta\s*\d+[-\.\s]?\d*',
        score=0.85
    )
    banco_recognizer = PatternRecognizer(
        supported_entity="DADOS_BANCARIOS",
        patterns=[banco_pattern],
        context=["banco", "conta", "agência", "corrente", "poupança", "bancário"]
    )
    recognizers.append(banco_recognizer)
    
    # Estado Civil - Padrão melhorado
    estado_civil_pattern = Pattern(
        name="estado_civil_pattern",
        regex=r'estado\s+civil\s*:?\s*(casado|solteiro|viúvo|divorciado|separado|união\s+estável)(\([a-z]\))?',
        score=0.8
    )
    estado_civil_recognizer = PatternRecognizer(
        supported_entity="ESTADO_CIVIL",
        patterns=[estado_civil_pattern],
        context=["estado civil", "casado", "solteiro", "viúvo", "divorciado", "marital"]
    )
    recognizers.append(estado_civil_recognizer)
    
    # Idade - Padrão melhorado
    idade_pattern = Pattern(
        name="idade_pattern",
        regex=r'\b(\d{1,3})\s+(anos|ano)(\s+de\s+idade)?\b|\bidade\s*:?\s*(\d{1,3})\s*(anos|ano)?\b',
        score=0.75
    )
    idade_recognizer = PatternRecognizer(
        supported_entity="IDADE",
        patterns=[idade_pattern],
        context=["idade", "anos", "aniversário", "idoso", "jovem", "adulto"]
    )
    recognizers.append(idade_recognizer)
    
    # Prontuário Médico
    prontuario_pattern = Pattern(
        name="prontuario_pattern",
        regex=r'prontuário\s*(médico|hospitalar)?\s*(n[°ºo.]?\s*\d+|:?\s*\d+)',
        score=0.8
    )
    prontuario_recognizer = PatternRecognizer(
        supported_entity="PRONTUARIO",
        patterns=[prontuario_pattern],
        context=["prontuário", "médico", "hospital", "clínica", "paciente", "saúde"]
    )
    recognizers.append(prontuario_recognizer)
    
    # Profissão - Padrão melhorado
    profissao_pattern = Pattern(
        name="profissao_pattern",
        regex=r'profissão\s*:?\s*[A-Za-zÀ-ÿ\s]+',
        score=0.7
    )
    profissao_recognizer = PatternRecognizer(
        supported_entity="PROFISSAO",
        patterns=[profissao_pattern],
        context=["trabalha", "emprego", "ocupação", "cargo", "ofício", "função"]
    )
    recognizers.append(profissao_recognizer)
    
    # Nacionalidade
    nacionalidade_pattern = Pattern(
        name="nacionalidade_pattern",
        regex=r'nacionalidade\s*:?\s*[A-Za-zÀ-ÿ]+',
        score=0.7
    )
    nacionalidade_recognizer = PatternRecognizer(
        supported_entity="NACIONALIDADE",
        patterns=[nacionalidade_pattern],
        context=["nacionalidade", "cidadão", "natural", "origem", "país"]
    )
    recognizers.append(nacionalidade_recognizer)
    
    return recognizers

def create_sensitive_terms_list():
    """
    Cria uma lista de termos sensíveis conforme a LGPD
    """
    termos_sensiveis = [
        # Raça e etnia
        "raça", "etnia", "cor da pele", "origem racial", "afrodescendente", 
        "indígena", "branco", "negro", "pardo", "amarelo", "ascendência", 
        "nacionalidade", "caucasiano", "negro", "afro-brasileiro", "preto",
        
        # Religião
        "religião", "crença", "fé", "igreja", "templo", "culto", "católico", 
        "evangélico", "protestante", "espírita", "candomblé", "umbanda", "ateu", 
        "agnóstico", "judaísmo", "islamismo", "budismo", "cristão", "muçulmano",
        "judeu", "hinduísta", "testemunha de jeová", "mórmon", "adventista",
        
        # Opinião política
        "opinião política", "partido político", "filiação partidária", "esquerda", 
        "direita", "centro", "conservador", "progressista", "liberal", "sindicalista", 
        "sindicato", "filiação sindical", "sindicalizado", "petista", "bolsonarista",
        "comunista", "socialista", "anarquista", "libertário", "monarquista",
        
        # Orientação sexual e identidade de gênero
        "orientação sexual", "heterossexual", "homossexual", "bissexual", "gay", 
        "lésbica", "transgênero", "LGBTQIA+", "vida sexual", "práticas sexuais",
        "assexual", "pansexual", "transexual", "cisgênero", "não-binário", 
        "gênero fluido", "queer", "travesti", "drag queen", "intersexual",
        
        # Saúde
        "saúde", "prontuário médico", "doença", "enfermidade", "diagnóstico", 
        "tratamento médico", "medicamento", "condição de saúde", "deficiência", 
        "transtorno", "histórico médico", "exame", "resultado de exame", "internação", 
        "cirurgia", "HIV", "AIDS", "câncer", "diabetes", "hipertensão",
        "depressão", "ansiedade", "esquizofrenia", "transtorno bipolar", "autismo",
        "epilepsia", "asma", "obesidade", "anorexia", "bulimia", "dependência química",
        "alcoolismo", "tabagismo", "hepatite", "tuberculose", "portador de necessidades",
        
        # Dados genéticos e biométricos
        "dados genéticos", "DNA", "genoma", "código genético", "material genético", 
        "dados biométricos", "impressão digital", "reconhecimento facial", "íris", 
        "retina", "voz", "assinatura", "marcha", "mapeamento genético",
        "exame de DNA", "teste de paternidade", "marcadores genéticos",
        
        # Antecedentes criminais
        "antecedentes criminais", "processo criminal", "histórico judicial", 
        "condenação", "delito", "crime", "contravenção", "ficha criminal",
        "processo penal", "inquérito policial", "prisão", "detenção", "liberdade condicional",
        "regime semi-aberto", "regime fechado", "foragido", "procurado"
    ]
    
    return termos_sensiveis

def create_custom_deny_list_recognizer(deny_list):
    """
    Cria um reconhecedor baseado em uma lista personalizada de palavras
    """
    # Verifica se a lista está vazia ou é None
    if not deny_list:
        return None
        
    # Se for uma string, tenta converter para lista
    if isinstance(deny_list, str):
        deny_list = [item.strip() for item in deny_list.split(",") if item.strip()]
    
    if not deny_list:
        return None
        
    patterns = []
    for i, word in enumerate(deny_list):
        if word and len(word) > 2:  # Ignora palavras muito curtas
            word_pattern = Pattern(
                name=f"custom_pattern_{i}",
                regex=r'\b' + re.escape(word) + r'\b',
                score=0.8
            )
            patterns.append(word_pattern)
            
    if patterns:
        return PatternRecognizer(
            supported_entity="PERSONALIZADO",
            patterns=patterns,
            context=[]
        )
    return None

def create_sensitive_terms_recognizer():
    """
    Cria um reconhecedor para os termos sensíveis da LGPD
    """
    sensitive_terms = create_sensitive_terms_list()
    patterns = []
    
    for i, term in enumerate(sensitive_terms):
        if term and len(term) > 2:  # Ignora termos muito curtos
            term_pattern = Pattern(
                name=f"sensitive_term_pattern_{i}",
                regex=r'\b' + re.escape(term) + r'\b',
                score=0.8
            )
            patterns.append(term_pattern)
            
    if patterns:
        return PatternRecognizer(
            supported_entity="TERMO_SENSIVEL",
            patterns=patterns,
            context=[]
        )
    return None

def setup_analyzer_engine(deny_list=None):
    """
    Configura o motor de análise do Presidio com os reconhecedores personalizados
    """
    try:
        # Criar um registro de reconhecedores
        registry = RecognizerRegistry()
        
        # Configurar o provedor de NLP se o spaCy estiver disponível
        if nlp:
            nlp_engine = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "pt", "model_name": "pt_core_news_md"}]
            }).create_engine()
        else:
            # Usar configuração mínima se o spaCy não estiver disponível
            nlp_engine = None
        
        # Adicionar os reconhecedores padrão da Presidio - carregue apenas os que funcionam bem com pt-BR
        registry.load_predefined_recognizers(languages=["pt"])
        
        # Adicionar os reconhecedores personalizados para pt-BR
        for recognizer in create_pt_br_recognizers():
            registry.add_recognizer(recognizer)
        
        # Adicionar reconhecedor de termos sensíveis
        sensitive_terms_recognizer = create_sensitive_terms_recognizer()
        if sensitive_terms_recognizer:
            registry.add_recognizer(sensitive_terms_recognizer)
        
        # Adicionar o reconhecedor personalizado de lista negada
        if deny_list:
            custom_deny_list_recognizer = create_custom_deny_list_recognizer(deny_list)
            if custom_deny_list_recognizer:
                registry.add_recognizer(custom_deny_list_recognizer)
        
        # Criar o motor de análise
        analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine
        )
        
        return analyzer
    
    except Exception as e:
        st.error(f"Erro ao configurar motor de análise: {str(e)}")
        return None

def setup_anonymizer_engine():
    """
    Configura o motor de anonimização do Presidio
    """
    try:
        return AnonymizerEngine()
    except Exception as e:
        st.error(f"Erro ao configurar motor de anonimização: {str(e)}")
        return None
    

def anonimizar_com_presidio(texto, deny_list=None, mascara='*'):
    """
    Anonimiza o texto usando o Presidio
    """
    if not texto:
        return texto, []
    
    try:
        # Configurar o motor de análise
        analyzer = setup_analyzer_engine(deny_list)
        if not analyzer:
            raise Exception("Não foi possível inicializar o motor de análise")
        
        # Configurar o motor de anonimização
        anonymizer = setup_anonymizer_engine()
        if not anonymizer:
            raise Exception("Não foi possível inicializar o motor de anonimização")
        
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
            "CPF": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": 11, "from_end": False}),
            "CNPJ": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": 14, "from_end": False}),
            "RG": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": 9, "from_end": False}),
            "EMAIL": OperatorConfig("mask", {"masking_char": mascara, "chars_to_mask": -1, "from_end": False}),
            "PERSONALIZADO": OperatorConfig("replace", {"new_value": mascara * 7}),
            "TERMO_SENSIVEL": OperatorConfig("replace", {"new_value": mascara * 7}),
        }
        
        # Converter os resultados do Presidio para o formato esperado
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
                st.warning(f"Erro na anonimização com Presidio: {str(e)}. Usando método de fallback.")
                # Fallback para método mais simples se o anonymizer falhar
                texto_anonimizado = texto
                for result in sorted(results, key=lambda x: x.start, reverse=True):
                    texto_anonimizado = (
                        texto_anonimizado[:result.start] + 
                        mascara * (result.end - result.start) + 
                        texto_anonimizado[result.end:]
                    )
        else:
            texto_anonimizado = texto
        
        return texto_anonimizado, achados
    
    except Exception as e:
        st.error(f"Erro ao usar Presidio: {str(e)}")
        # Em caso de erro, retornar o texto original e lista vazia de achados
        return texto, []
    
def anonimizar_simples(texto, deny_list=None, mascara='*'):
    """
    Função de anonimização usando expressões regulares
    """
    if not texto:
        return texto, []
    
    # Lista para armazenar os achados
    achados = []
    texto_original = texto
    
    # Preparar a lista de termos sensíveis
    termos_sensiveis = create_sensitive_terms_list()
    
    # Preparar a lista de palavras personalizadas
    palavras_personalizadas = []
    if deny_list:
        if isinstance(deny_list, str):
            palavras_personalizadas = [p.strip() for p in deny_list.split(",") if p.strip()]
        else:
            palavras_personalizadas = deny_list
    
    # Padrões para identificar informações pessoais - melhorados para mais precisão
    padroes = {
        'CPF': r'\b\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\.\s]?\d{2}\b',
        'CNPJ': r'\b\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[/\.\s]?\d{4}[-\.\s]?\d{2}\b',
        'RG': r'\b\d{1,2}[\.\s]?\d{3}[\.\s]?\d{3}[-\.\s]?[\dxX]\b|\b[A-Za-z]{2}[-\.\s]?\d{6,8}\b',
        'TELEFONE': r'\(\s*\d{2}\s*\)\s*9?\s*\d{4}[-\.\s]?\d{4}|\b\d{2}\s*9?\s*\d{4}[-\.\s]?\d{4}\b|\b\d{5}[-\.\s]?\d{4}\b',
        'CEP': r'\b\d{5}[-\.\s]?\d{3}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|[Ee]-?mail:?\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        'CARTAO_CREDITO': r'\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b',
        'DATA': r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        'ENDERECO_RUA': r'\b(rua|avenida|av\.|alameda|praça|travessa|rod\.|rodovia|estrada|quadra|lote)\s+[A-Za-zÀ-ÿ\s\.\,0-9]+(,?\s*n[º°\.]\s*\d+)?(\s*[-,]\s*[A-Za-zÀ-ÿ\s\.\,0-9]+)?\b',
        'ENDERECO_COMPLETO': r'residente\s+(e\s+domiciliado\s+)?(a|à|na|no)?\s+.{5,50}(,\s+n°\.?\s+\d+)?|morador\s+(a|à|na|no)?\s+.{5,50}(,\s+n°\.?\s+\d+)?',
        'ENDERECO_BAIRRO': r'\b(bairro|b\.)\s+[A-Za-zÀ-ÿ\s]+\b',
        'ENDERECO_CIDADE': r'\b(cidade|cid\.|município) de\s+[A-Za-zÀ-ÿ\s]+\b|[Ee]m\s+[A-Z][a-zÀ-ÿ]+(/[A-Z]{2})?\b',
        'ESTADO_CIVIL': r'\bestado\s+civil\s*:?\s*(casado|solteiro|viúvo|divorciado|separado|união estável)(\([a-z]\))?\b',
        'PROFISSAO': r'\bprofissão\s*:?\s*[A-Za-zÀ-ÿ\s]+\b',
        'FILIACAO': r'\bfilho\s+(de|da|do)\s+[A-Za-zÀ-ÿ\s]+(\s+e\s+de\s+[A-Za-zÀ-ÿ\s]+)?\b',
        'DATA_NASCIMENTO': r'\bnascido\s+em\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\bnascida\s+em\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\bdata\s+de\s+nascimento:?\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}\b',
        'IDADE': r'\b(\d{1,3})\s+(anos|ano)(\s+de\s+idade)?\b|\bidade\s*:?\s*(\d{1,3})\s*(anos|ano)?\b',
        'PIS_PASEP': r'\b\d{3}\.?\d{5}\.?\d{2}-?\d{1}\b',
        'TITULO_ELEITOR': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        'CNH': r'\bcnh\s*:?\s*\d{9,11}\b|\bcarteira\s+de\s+habilitação\s*:?\s*\d{9,11}\b',
        'BANCO': r'\b(banco|agência|ag[\.:]?|conta[\.:]?)\s*\d+[-\.\s]?\d*\b|\bagência\s*\d+[-\.\s]?\d*\s*,?\s*conta\s*\d+[-\.\s]?\d*\b',
        'NOME_EMPRESA': r'\b([A-Z][A-ZÀ-Ú]+\s+){2,}([A-Z][A-ZÀ-Ú\s]*)+\b',
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
                    
                    # Substituir a informação por máscara
                    texto = texto[:start] + mascara * len(info) + texto[end:]
        # Tratamento especial para endereços e outros padrões longos
        elif tipo in ['ENDERECO_RUA', 'ENDERECO_COMPLETO', 'PROFISSAO', 'FILIACAO']:
            for match in re.finditer(padrao, texto, re.IGNORECASE | re.UNICODE):
                info = match.group()
                start = match.start()
                end = match.end()
                
                # Verificar se é um achado relevante (evitar falsos positivos)
                if len(info) >= 5:
                    # Armazenar o achado
                    achados.append({
                        'Tipo de Entidade': tipo,
                        'Texto': info,
                        'Início': start,
                        'Fim': end,
                        'Confiança': 0.9
                    })
                    
                    # Substituir a informação por máscara
                    texto = texto[:start] + mascara * len(info) + texto[end:]
        else:
            for match in re.finditer(padrao, texto, re.IGNORECASE | re.UNICODE):
                info = match.group()
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
                
                # Substituir a informação por máscara
                texto = texto[:start] + mascara * len(info) + texto[end:]

# Procurar e substituir nomes comuns brasileiros (sobrenomes e nomes próprios)
    nomes_comuns = [
        # Sobrenomes mais comuns
        "Silva", "Santos", "Oliveira", "Souza", "Lima", "Pereira", "Ferreira", 
        "Almeida", "Costa", "Rodrigues", "Gomes", "Martins", "Araújo", "Carvalho",
        "Ribeiro", "Alves", "Monteiro", "Mendes", "Barros", "Freitas", "Barbosa",
        "Pinto", "Moura", "Cavalcanti", "Dias", "Castro", "Campos", "Cardoso",
        
        # Nomes próprios masculinos comuns
        "João", "José", "Antonio", "Carlos", "Paulo", "Pedro", "Lucas", "Marcos", "Luis",
        "Gabriel", "Rafael", "Daniel", "Marcelo", "Bruno", "Eduardo", "Felipe", "Raimundo",
        "Francisco", "Jorge", "Roberto", "Manoel", "Ricardo", "Diego", "André", "Fernando",
        
        # Nomes próprios femininos comuns
        "Maria", "Ana", "Francisca", "Antonia", "Adriana", "Juliana", "Marcia", "Fernanda",
        "Patricia", "Aline", "Sandra", "Camila", "Amanda", "Bruna", "Jessica", "Leticia",
        "Julia", "Beatriz", "Luciana", "Márcia", "Lúcia", "Cristina", "Débora", "Cláudia",
        "Bárbara", "Bianca", "Carla", "Carolina", "Cecília", "Diana", "Eliana"
    ]
    
    for nome in nomes_comuns:
        # Usando regex para encontrar nomes como palavras inteiras (não parte de outras palavras)
        padrao_nome = r'\b' + re.escape(nome) + r'\b'
        for match in re.finditer(padrao_nome, texto, re.IGNORECASE | re.UNICODE):
            info = match.group()
            start = match.start()
            end = match.end()
            
            # Verificar o contexto (apenas nomes precedidos ou seguidos por outro nome/sobrenome)
            # Isso reduz falsos positivos para palavras que são nomes mas também são comuns no português
            contexto_valido = False
            
            # Verifica se há outro nome ou sobrenome antes ou depois (até 25 caracteres de distância)
            texto_antes = texto[max(0, start-25):start]
            texto_depois = texto[end:min(len(texto), end+25)]
            
            for outro_nome in nomes_comuns:
                if outro_nome != nome:
                    if re.search(r'\b' + re.escape(outro_nome) + r'\b', texto_antes, re.IGNORECASE | re.UNICODE) or \
                       re.search(r'\b' + re.escape(outro_nome) + r'\b', texto_depois, re.IGNORECASE | re.UNICODE):
                        contexto_valido = True
                        break
            
            # Verificar se o nome está precedido por Sr., Dr., etc.
            prefixos = ["Sr.", "Sra.", "Dr.", "Dra.", "Srta.", "Prof.", "Profa.", "Doutor", "Doutora"]
            for prefixo in prefixos:
                if re.search(r'\b' + re.escape(prefixo) + r'\s+$', texto_antes, re.IGNORECASE | re.UNICODE):
                    contexto_valido = True
                    break
            
            if contexto_valido:
                # Armazenar o achado
                achados.append({
                    'Tipo de Entidade': 'NOME',
                    'Texto': info,
                    'Início': start,
                    'Fim': end,
                    'Confiança': 0.85
                })
                
                # Substituir a informação por máscara
                texto = texto[:start] + mascara * len(info) + texto[end:]

# Procurar nomes completos no formato "Nome Sobrenome"
    # Este padrão é mais restritivo para evitar falsos positivos
    # Exemplo: "João Silva" ou "Maria Oliveira Santos"
    padrao_nome_completo = r'\b[A-Z][a-zÀ-ÿ]+\s+([A-Z][a-zÀ-ÿ]+\s+){1,2}[A-Z][a-zÀ-ÿ]+\b'
    for match in re.finditer(padrao_nome_completo, texto, re.UNICODE):
        info = match.group()
        start = match.start()
        end = match.end()
        
        # Verificar se tem pelo menos 2 palavras e 10 caracteres para ser um nome completo válido
        palavras = [p for p in info.split() if p.strip()]
        if len(palavras) >= 2 and len(info) >= 10:
            # Armazenar o achado
            achados.append({
                'Tipo de Entidade': 'NOME_COMPLETO',
                'Texto': info,
                'Início': start,
                'Fim': end,
                'Confiança': 0.9
            })
            
            # Substituir a informação por máscara
            texto = texto[:start] + mascara * len(info) + texto[end:]
    
    # Após os reconhecedores padrão, aplicar os termos sensíveis
    for termo in termos_sensiveis:
        padrao_termo = r'\b' + re.escape(termo) + r'\b'
        for match in re.finditer(padrao_termo, texto, re.IGNORECASE | re.UNICODE):
            info = match.group()
            start = match.start()
            end = match.end()
            
            # Armazenar o achado
            achados.append({
                'Tipo de Entidade': 'TERMO_SENSIVEL',
                'Texto': info,
                'Início': start,
                'Fim': end,
                'Confiança': 0.9
            })
            
            # Substituir a informação por máscara
            texto = texto[:start] + mascara * len(info) + texto[end:]
    
    # Adicionar palavras personalizadas
    if palavras_personalizadas:
        for palavra in palavras_personalizadas:
            if palavra and len(palavra) > 2:  # Evitar palavras muito curtas
                padrao_palavra = r'\b' + re.escape(palavra) + r'\b'
                for match in re.finditer(padrao_palavra, texto, re.IGNORECASE | re.UNICODE):
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
                    
                    # Substituir a informação por máscara
                    texto = texto[:start] + mascara * len(info) + texto[end:]
    
    return texto, achados

def anonimizar_hibrido(texto, palavras_adicionais=None, mascara='*', tolerancia=0.5):
    """
    Função híbrida que combina o Presidio com o método regex para melhor cobertura
    """
    if not texto:
        return texto, []
    
    # Preparar a lista de palavras adicionais
    deny_list = []
    if palavras_adicionais:
        if isinstance(palavras_adicionais, str):
            deny_list = [p.strip() for p in palavras_adicionais.split(",") if p.strip()]
        else:
            deny_list = palavras_adicionais
    
    start_time = time.time()
    # Primeiro, usar Presidio para anonimização
    texto_presidio, achados_presidio = anonimizar_com_presidio(texto, deny_list, mascara)
    presidio_time = time.time() - start_time
    
    # Em seguida, usar o método de expressões regulares
    start_time = time.time()
    texto_regex, achados_regex = anonimizar_simples(texto, deny_list, mascara)
    regex_time = time.time() - start_time
    
    # Avaliar qual método encontrou mais dados sensíveis
    # Determinar melhor estratégia:
    # 1. Se o Presidio encontrou muitos dados (mais de 30% a mais que o regex), usar Presidio
    # 2. Se o regex encontrou muitos dados (mais de 30% a mais que o Presidio), usar regex
    # 3. Caso contrário, combinar os resultados para máxima cobertura
    
    if len(achados_presidio) > len(achados_regex) * 1.3:
        # Presidio encontrou significativamente mais
        return texto_presidio, achados_presidio
    
    elif len(achados_regex) > len(achados_presidio) * 1.3:
        # Regex encontrou significativamente mais
        return texto_regex, achados_regex
    
    else:
        # Combinar os resultados para máxima cobertura
        # Primeiro, criar um conjunto de spans para evitar sobrepor anonimizações
        # (chave: (inicio, fim), valor: achado)
        todos_spans = {}
        
        # Adicionar spans do Presidio
        for achado in achados_presidio:
            span = (achado['Início'], achado['Fim'])
            # Se já existe um span sobreposto, manter apenas o de maior confiança
            for sp in list(todos_spans.keys()):
                if (span[0] <= sp[1] and span[1] >= sp[0]):  # Se há sobreposição
                    if achado['Confiança'] > todos_spans[sp]['Confiança']:
                        del todos_spans[sp]
                        todos_spans[span] = achado
                    break
            else:
                todos_spans[span] = achado
        
        # Adicionar spans do Regex, sem sobrepor os do Presidio
        for achado in achados_regex:
            span = (achado['Início'], achado['Fim'])
            sobreposto = False
            # Verificar se este span se sobrepõe a algum já adicionado
            for sp in list(todos_spans.keys()):
                if (span[0] <= sp[1] and span[1] >= sp[0]):  # Se há sobreposição
                    sobreposto = True
                    # Se tiver maior confiança, substituir o existente
                    if achado['Confiança'] > todos_spans[sp]['Confiança']:
                        del todos_spans[sp]
                        todos_spans[span] = achado
                    break
            
            if not sobreposto:
                todos_spans[span] = achado
        
        # Ordenar spans por posição (início, fim) para garantir anonimização correta
        spans_ordenados = sorted(todos_spans.keys(), reverse=True)
        
        # Anonimizar o texto original usando todos os spans encontrados
        texto_final = texto
        for span in spans_ordenados:
            inicio, fim = span
            texto_final = texto_final[:inicio] + mascara * (fim - inicio) + texto_final[fim:]
        
        # Criar lista de achados final
        achados_final = [todos_spans[span] for span in sorted(todos_spans.keys())]
        
        return texto_final, achados_final
    
def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF"""
    try:
        reader = PdfReader(pdf_file)
        texto = ""
        total_pages = len(reader.pages)
        
        # Para PDFs muito grandes, mostrar progresso
        if total_pages > 10:
            progress_bar = st.progress(0)
            for i, page in enumerate(reader.pages):
                texto += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / total_pages)
            progress_bar.empty()
        else:
            for page in reader.pages:
                texto += page.extract_text() + "\n"
                
        if not texto.strip():
            st.warning("O PDF parece estar vazio ou contém apenas imagens. A extração de texto pode não funcionar corretamente.")
            
        return texto
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return ""

def extract_text_from_csv(csv_file):
    """Extrai texto de um arquivo CSV"""
    try:
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Erro ao ler CSV com encoding {encoding}: {str(e)}")
                
        if df is None:
            try:
                # Último recurso: tentar detectar o delimitador
                df = pd.read_csv(csv_file, encoding='latin-1', sep=None, engine='python')
            except Exception as e:
                st.error(f"Erro ao ler CSV: {str(e)}")
                return ""
        
        # Para CSVs muito grandes, melhor apresentar resumo do que o conteúdo inteiro
        if len(df) > 1000 or len(df.columns) > 20:
            st.warning("O CSV é muito grande, extraindo apenas cabeçalhos e amostra para anonimização.")
            texto = df.head(100).to_string() + "\n...\n" + df.tail(100).to_string()
        else:
            texto = df.to_string()
            
        return texto
    except Exception as e:
        st.error(f"Erro ao ler CSV: {str(e)}")
        return ""
    
def process_file(uploaded_file, tolerancia, palavras, mascara):
    """Processa um arquivo enviado pelo usuário"""
    try:
        # Mostrar spinner enquanto processa
        with st.spinner(f"Processando arquivo {uploaded_file.name}..."):
            if uploaded_file.type == "application/pdf" or ".pdf" in uploaded_file.name.lower():
                texto = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/csv" or ".csv" in uploaded_file.name.lower():
                texto = extract_text_from_csv(uploaded_file)
            else:
                st.error("Tipo de arquivo não suportado. Utilize PDF ou CSV.")
                return None
            
            if not texto or len(texto.strip()) < 10:
                st.error("Não foi possível extrair texto do arquivo ou o arquivo está vazio.")
                return None
            
            # Para arquivos muito grandes, mostrar aviso e limitar tamanho
            if len(texto) > 500000:  # Limitar arquivos muito grandes (500K caracteres)
                st.warning(f"Arquivo muito grande ({len(texto)} caracteres). Processando apenas os primeiros 500.000 caracteres.")
                texto = texto[:500000] + "\n[...Texto truncado devido ao tamanho...]"
            
            return process_text(texto, tolerancia, palavras, mascara)
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None

def process_text(texto, tolerancia, palavras, mascara):
    """Processa um texto para anonimização, usando abordagem híbrida (Presidio + Regex)"""
    try:
        # Verificar se o texto está em branco
        if not texto or len(texto.strip()) < 10:
            return {
                "texto": texto,
                "findings": [],
                "message": "O texto está vazio ou é muito curto para processamento."
            }
        
        # Verificar modo de anonimização selecionado
        modo = st.session_state.get("modo_anonimizacao", "Automático (recomendado)")
        
        # Mostrar seleção atual ao usuário
        st.info(f"Usando modo de anonimização: {modo}")
        
        # Preparar a lista de palavras a serem negadas
        deny_list = []
        if palavras:
            deny_list = [p.strip() for p in palavras.split(",") if p.strip()]
            if deny_list:
                st.info(f"Palavras adicionais para detecção: {', '.join(deny_list)}")
        
        tempo_inicio = time.time()
        
        # Escolher função de anonimização com base no modo selecionado
        if modo == "Presidio":
            texto_anonimizado, achados = anonimizar_com_presidio(texto, deny_list, mascara)
        elif modo == "Regex":
            texto_anonimizado, achados = anonimizar_simples(texto, deny_list, mascara)
        else:  # Automático (híbrido)
            texto_anonimizado, achados = anonimizar_hibrido(texto, palavras, mascara, tolerancia)
        
        tempo_total = time.time() - tempo_inicio
        
        # Se não encontrou nada para anonimizar
        if not achados:
            # Retornar o texto original com uma nota
            return {
                "texto": texto,
                "findings": [],
                "message": "Nenhuma informação pessoal foi identificada no texto.",
                "tempo": tempo_total
            }
        
        # Preparar a resposta
        response = {
            "texto": texto_anonimizado,
            "findings": achados,
            "tempo": tempo_total
        }
        
        return response
        
    except Exception as e:
        st.error(f"Erro ao anonimizar texto: {str(e)}")
        traceback.print_exc()
        return None
    
def main():
    st.title("Anonimizador de Textos - LGPD")
    st.markdown("""
    ### Proteja dados sensíveis de acordo com a Lei Geral de Proteção de Dados
    Esta ferramenta ajuda a identificar e mascarar informações pessoais e sensíveis em textos,
    utilizando técnicas avançadas de anonimização otimizadas para o português brasileiro.
    """)
    
    # Inicialização do modelo spaCy e verificação
    if nlp is None:
        st.warning("""
        ⚠️ O modelo de linguagem spaCy não foi carregado completamente. 
        A ferramenta funcionará, mas com precisão reduzida na detecção de entidades.
        """)
    
    # Opções de configuração
    with st.expander("Configurações de Anonimização", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            tolerancia = st.slider(
                "Tolerância de detecção", 
                min_value=0.3, 
                max_value=0.9, 
                value=0.5,
                step=0.05,
                help="Valores menores detectam mais padrões, mas podem gerar falsos positivos"
            )
        with col2:
            palavras = st.text_input(
                "Palavras adicionais a mascarar", 
                placeholder="Palavras separadas por vírgula",
                help="Lista de palavras específicas que devem ser detectadas e anonimizadas"
            )
        with col3:
            mascara = st.text_input(
                "Caractere de máscara",
                value="*",
                max_chars=1,
                help="Caractere usado para substituir informações pessoais"
            )
            if not mascara:
                mascara = "*"  # Garantir que sempre há um caractere de máscara
            
        # Modo de anonimização
        st.radio(
            "Modo de anonimização",
            ["Automático (recomendado)", "Presidio", "Regex"],
            index=0,
            key="modo_anonimizacao",
            help="Escolha o método de detecção de informações pessoais. O modo Automático combina ambas as técnicas para melhor cobertura."
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
        
        col1, col2 = st.columns([3, 1])
        with col1:
            btn_anon = st.button("Anonimizar Texto", type="primary", key="btn_texto", use_container_width=True)
        with col2:
            btn_limpar = st.button("Limpar", key="btn_limpar", use_container_width=True)
            
        if btn_limpar:
            # Limpar o texto de entrada
            st.session_state['user_input'] = ""
            st.experimental_rerun()
        
        if btn_anon:
            if user_input:
                with st.spinner("Processando texto..."):
                    resultado = process_text(user_input, tolerancia, palavras, mascara)
                    if resultado:
                        if "message" in resultado:
                            st.info(resultado["message"])
                            if "tempo" in resultado:
                                st.info(f"Tempo de processamento: {resultado['tempo']:.2f} segundos")
                        else:
                            st.success(f"Texto anonimizado com sucesso em {resultado.get('tempo', 0):.2f} segundos!")
                        
                        # Exibir os achados
                        if "findings" in resultado and resultado["findings"]:
                            with st.expander("Informações detectadas", expanded=True):
                                findings_df = pd.DataFrame(resultado["findings"])
                                # Ordenar por confiança (decrescente)
                                findings_df = findings_df.sort_values(by='Confiança', ascending=False)
                                st.dataframe(findings_df, use_container_width=True)
                                
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
                                
                                # Distribuição por tipo de entidade
                                tipo_count = findings_df['Tipo de Entidade'].value_counts()
                                st.subheader("Distribuição por tipo de dados")
                                st.bar_chart(tipo_count)

# Exibir o texto anonimizado
                        st.subheader("Resultado:")
                        st.markdown(
                            f"""<div style="padding: 15px; border-radius: 5px; background-color: #f0f2f6; white-space: pre-wrap; word-wrap: break-word;">
                            {resultado["texto"]}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        
                        # Opção de download
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Baixar texto anonimizado",
                                data=resultado["texto"],
                                file_name="texto_anonimizado.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        # Opção para relatório
                        if "findings" in resultado and resultado["findings"]:
                            with col2:
                                report = generate_anonymization_report(user_input, resultado)
                                st.download_button(
                                    label="Baixar relatório detalhado",
                                    data=report,
                                    file_name="relatorio_anonimizacao.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
            else:
                st.warning("Por favor, insira um texto para anonimizar.")

    with tab2:
            st.subheader("Anonimização de Arquivo")
            uploaded_file = st.file_uploader("Faça upload do arquivo:", type=["pdf", "csv"], key="file_uploader")
            
            if uploaded_file is not None:
                # Exibir detalhes do arquivo
                file_details = {"Nome": uploaded_file.name, "Tipo": uploaded_file.type, "Tamanho": f"{uploaded_file.size/1024:.2f} KB"}
                
                # Criar uma tabela para melhor visualização
                st.markdown("### Detalhes do Arquivo")
                details_df = pd.DataFrame([file_details])
                st.dataframe(details_df, use_container_width=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    btn_anon_file = st.button("Anonimizar Arquivo", type="primary", key="btn_arquivo", use_container_width=True)
                with col2:
                    btn_limpar_file = st.button("Limpar", key="btn_limpar_file", use_container_width=True)
                    
                if btn_limpar_file:
                    # Limpar o arquivo
                    st.session_state['file_uploader'] = None
                    st.experimental_rerun()
                
                if btn_anon_file:
                    resultado = process_file(uploaded_file, tolerancia, palavras, mascara)
                    if resultado:
                        if "message" in resultado:
                            st.info(resultado["message"])
                            if "tempo" in resultado:
                                st.info(f"Tempo de processamento: {resultado['tempo']:.2f} segundos")
                        else:
                            st.success(f"Arquivo anonimizado com sucesso em {resultado.get('tempo', 0):.2f} segundos!")
                        
                        # Exibir os achados
                        if "findings" in resultado and resultado["findings"]:
                            with st.expander("Informações detectadas", expanded=True):
                                findings_df = pd.DataFrame(resultado["findings"])
                                # Ordenar por confiança (decrescente)
                                findings_df = findings_df.sort_values(by='Confiança', ascending=False)
                                st.dataframe(findings_df, use_container_width=True)
                                
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
                                
                                # Distribuição por tipo de entidade
                                tipo_count = findings_df['Tipo de Entidade'].value_counts()
                                st.subheader("Distribuição por tipo de dados")
                                st.bar_chart(tipo_count)

    # Exibir o texto anonimizado
                        st.subheader("Resultado:")
                        st.text_area("Texto anonimizado", value=resultado["texto"], height=400)
                        
                        # Opções de download
                        col1, col2 = st.columns(2)
                        with col1:
                            text_download = resultado["texto"]
                            st.download_button(
                                label="Baixar texto anonimizado",
                                data=text_download,
                                file_name=f"{uploaded_file.name}_anonimizado.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        # Opção para relatório
                        if "findings" in resultado and resultado["findings"]:
                            with col2:
                                report = generate_anonymization_report(
                                    f"Arquivo: {uploaded_file.name}", 
                                    resultado
                                )
                                st.download_button(
                                    label="Baixar relatório detalhado",
                                    data=report,
                                    file_name=f"{uploaded_file.name}_relatorio.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )


def generate_anonymization_report(original_text, resultado):
    """Gera um relatório detalhado da anonimização"""
    findings = resultado.get("findings", [])
    texto_anonimizado = resultado.get("texto", "")
    tempo_processamento = resultado.get("tempo", 0)
    
    report = "RELATÓRIO DE ANONIMIZAÇÃO DE DADOS\n"
    report += "=" * 60 + "\n\n"
    
    # Data e hora
    report += f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    report += f"Tempo de processamento: {tempo_processamento:.2f} segundos\n\n"
    
    # Resumo
    report += "RESUMO:\n"
    report += "-" * 60 + "\n"
    report += f"Total de informações sensíveis encontradas: {len(findings)}\n"
    
    if findings:
        # Contar tipos de entidades
        tipos_entidade = {}
        for finding in findings:
            tipo = finding.get('Tipo de Entidade', 'Desconhecido')
            tipos_entidade[tipo] = tipos_entidade.get(tipo, 0) + 1
        
        report += "Distribuição por tipo de informação sensível:\n"
        for tipo, contagem in sorted(tipos_entidade.items(), key=lambda x: x[1], reverse=True):
            report += f"  - {tipo}: {contagem} ocorrência(s)\n"
        
        # Confiança média
        confianca_total = sum(finding.get('Confiança', 0) for finding in findings)
        confianca_media = confianca_total / len(findings)
        report += f"Confiança média: {confianca_media:.2f}\n\n"
        
        # Detalhes das informações encontradas
        report += "DETALHES DAS INFORMAÇÕES ENCONTRADAS:\n"
        report += "-" * 60 + "\n"
        
        # Ordenar por tipo de entidade para facilitar a leitura
        findings_sorted = sorted(findings, key=lambda x: (x.get('Tipo de Entidade', ''), -x.get('Confiança', 0)))
        
        for i, finding in enumerate(findings_sorted, 1):
            report += f"Item {i}:\n"
            report += f"  Tipo: {finding.get('Tipo de Entidade', 'Desconhecido')}\n"
            report += f"  Texto: {finding.get('Texto', '')}\n"
            report += f"  Confiança: {finding.get('Confiança', 0):.2f}\n"
            report += f"  Posição: {finding.get('Início', 0)}-{finding.get('Fim', 0)}\n"
            report += "\n"
    
    # Amostras de texto (original e anonimizado)
    # Limitando para evitar relatórios muito grandes
    max_sample_length = 1500
    sample_original = original_text[:max_sample_length] + ("..." if len(original_text) > max_sample_length else "")
    sample_anon = texto_anonimizado[:max_sample_length] + ("..." if len(texto_anonimizado) > max_sample_length else "")
    
    report += "AMOSTRA DE TEXTO ORIGINAL:\n"
    report += "-" * 60 + "\n"
    report += sample_original + "\n\n"
    
    report += "AMOSTRA DE TEXTO ANONIMIZADO:\n"
    report += "-" * 60 + "\n"
    report += sample_anon + "\n\n"
    
    # Recomendações
    report += "RECOMENDAÇÕES E MELHORES PRÁTICAS:\n"
    report += "-" * 60 + "\n"
    report += "1. Verifique manualmente o texto anonimizado para garantir que todas as informações sensíveis foram detectadas.\n"
    
    # Recomendações específicas baseadas na quantidade de dados encontrados
    if not findings:
        report += "2. Não foram encontradas informações sensíveis. Considere reduzir a tolerância de detecção se você esperava encontrar dados.\n"
    elif len(findings) < 5:
        report += "2. Foram encontradas poucas informações sensíveis. Considere reduzir a tolerância de detecção para aumentar a sensibilidade.\n"
    elif len(findings) > 100:
        report += "2. Foi detectado um grande volume de informações sensíveis. Verifique a ocorrência de falsos positivos.\n"
    
    report += "3. Para casos específicos, adicione palavras personalizadas à lista de detecção.\n"
    report += "4. Lembre-se que esta é uma ferramenta de auxílio e não substitui uma revisão manual cuidadosa.\n"
    report += "5. Sempre armazene e processe o texto original de acordo com as recomendações de segurança da LGPD.\n\n"
    
    report += "=" * 60 + "\n"
    report += "Este relatório foi gerado automaticamente pelo Anonimizador de Textos - LGPD.\n"
    report += f"Relatório criado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n"
    
    return report

if __name__ == "__main__":
    main()
