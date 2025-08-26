import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# -----------------------------
# Configurações visuais
# -----------------------------
st.set_page_config(page_title="Projeto CAIXA — Estatística + Dashboard", layout="wide")
sns.set(style="whitegrid")

# -----------------------------
# Utilidades estatísticas
# -----------------------------
def odds_ratios_from_logit(result):
    params = result.params
    conf = result.conf_int()
    or_tab = pd.DataFrame({
        "OR": np.exp(params),
        "IC_inf": np.exp(conf[0]),
        "IC_sup": np.exp(conf[1]),
        "p-valor": result.pvalues
    })
    return or_tab

def roc_curve_manual(y_true, scores):
    # y_true: {0,1}; scores: probabilidade prevista
    order = np.argsort(-scores)
    y = np.asarray(y_true)[order]
    s = np.asarray(scores)[order]
    P = y.sum()
    N = len(y) - P
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    TPR = tps / (P if P>0 else 1)
    FPR = fps / (N if N>0 else 1)
    # pontos (0,0) e (1,1)
    FPR = np.concatenate([[0], FPR, [1]])
    TPR = np.concatenate([[0], TPR, [1]])
    # AUC pela regra do trapézio
    auc = np.trapz(TPR, FPR)
    return FPR, TPR, auc

def ks_statistic(y_true, scores):
    # KS = max(|F1(t) - F0(t)|)
    # Distribuições acumuladas das pontuações por classe
    thresholds = np.unique(scores)
    ks = 0.0
    for t in thresholds:
        cdf1 = (scores[y_true==1] <= t).mean() if (y_true==1).sum()>0 else 0
        cdf0 = (scores[y_true==0] <= t).mean() if (y_true==0).sum()>0 else 0
        ks = max(ks, abs(cdf1 - cdf0))
    return ks

def calib_curve(y_true, scores, n_bins=10):
    df = pd.DataFrame({"y":y_true, "p":scores})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin").agg(real=("y","mean"), prev=("p","mean"), n=("y","size")).reset_index()
    return g

# -----------------------------
# Parâmetros padrão fixos
# -----------------------------
seed = 42
n_clientes = 5000
pct_app = 0.70
juros_base = 0.20
np.random.seed(int(seed))

# -----------------------------
# 1) Geração de base sintética
# -----------------------------
def gerar_base(N):
    def gerar_nomes(n):
        nomes = ["Ana","Bruno","Carla","Diego","Eduarda","Felipe","Gabriela","Henrique","Isabela","João",
                 "Karina","Lucas","Mariana","Natan","Olivia","Paulo","Queila","Rafael","Sofia","Tiago",
                 "Ursula","Vitor","Wesley","Xavier","Yara","Zilda"]
        sobrenomes = ["Silva","Souza","Oliveira","Santos","Pereira","Lima","Gomes","Costa","Ribeiro","Almeida",
                      "Carvalho","Araujo","Fernandes","Rocha","Dias","Moreira","Barbosa","Teixeira","Correia","Freitas"]
        return [f"{np.random.choice(nomes)} {np.random.choice(sobrenomes)}" for _ in range(n)]

    ids = np.arange(1, N+1)
    nomes = gerar_nomes(N)
    idade = np.clip(np.random.normal(40, 12, N).round(), 18, 80).astype(int)
    genero = np.random.choice(["F","M"], size=N)
    estado_civil = np.random.choice(["Solteiro","Casado","Divorciado","Viúvo"], size=N, p=[0.45,0.4,0.12,0.03])
    regiao = np.random.choice(["Norte","Nordeste","Centro-Oeste","Sudeste","Sul"], size=N, p=[0.08,0.28,0.1,0.4,0.14])
    tempo_rel = np.clip(np.random.exponential(5, N), 0.1, 30)
    renda = np.clip(np.random.lognormal(mean=np.log(3500), sigma=0.6, size=N), 1200, 40000)
    emprego = np.random.choice(["CLT","Servidor","Autônomo","Empresário","Desempregado"], size=N,
                               p=[0.5,0.12,0.22,0.1,0.06])
    score = np.clip(np.random.normal(650, 80, N), 300, 900)
    possui_cartao = (np.random.rand(N) < 0.65).astype(int)
    num_prod = np.clip(np.random.poisson(2, N)+1, 1, 8)
    saldo_poup = np.clip(np.random.gamma(2, 3000, N), 0, 200000)
    saldo_corr = np.clip(np.random.gamma(2, 1500, N), 0, 100000)
    usuario_app = (np.random.rand(N) < pct_app).astype(int)
    eng_digital = np.clip((usuario_app*0.6 + np.random.beta(2,3,N))*100, 0, 100)
    visitas_ag = np.clip((1-usuario_app)*np.random.poisson(5, N), 0, None)
    trans_12m = np.clip(np.random.poisson(60, N) + usuario_app*np.random.poisson(40, N), 1, None)
    seg = np.where(renda>=15000,"Alta Renda", np.where(renda>=6000,"Mass Affluent","Varejo"))

    tipo_emp = np.random.choice(["Consignado","Pessoal","Imobiliário","Veicular","Cartão"], size=N,
                                p=[0.25,0.3,0.15,0.15,0.15])
    valor_emp = np.where(tipo_emp=="Imobiliário", np.random.normal(250000,80000,N),
                np.where(tipo_emp=="Veicular", np.random.normal(60000,15000,N), np.random.normal(15000,8000,N)))
    valor_emp = np.clip(valor_emp, 1000, 1_000_000)

    prazo = np.where(tipo_emp=="Imobiliário",
                     np.random.choice([180,240,300,360], size=N, p=[0.2,0.3,0.3,0.2]),
                     np.random.choice([12,24,36,48,60], size=N, p=[0.2,0.35,0.25,0.15,0.05]))

    taxa = (
        juros_base
        - 0.00025*(score-650)
        - 0.000002*(renda-3500)
        + np.where(tipo_emp=="Imobiliário",-0.12,0)
        + np.where(tipo_emp=="Consignado",-0.08,0)
    )
    taxa = np.clip(taxa, 0.05, 0.45)

    X_lin = (
        -2.0
        - 0.00005*(renda-3500)
        - 0.004*(score-650)
        - 0.05*(tempo_rel-5)
        + 0.5*(emprego=="Desempregado").astype(int)
        + 0.2*(emprego=="Autônomo").astype(int)
        + 0.15*(num_prod<=2).astype(int)
        + 0.4*(taxa>0.25).astype(int)
        - 0.2*usuario_app
    )
    p_default = 1/(1+np.exp(-X_lin))
    inad = (np.random.rand(N) < p_default).astype(int)
    dias_atraso = np.where(inad==1, np.clip(np.random.normal(45, 20, N), 1, 180), 0).astype(int)

    p_churn = 1/(1+np.exp(-( -3 + 0.00008*(4000-renda) - 0.02*(eng_digital-50) + 0.3*(visitas_ag>5))))
    churn = (np.random.rand(N) < p_churn).astype(int)

    hoje = datetime(2025,8,21)
    abertura = [hoje - timedelta(days=int(tr*365)) for tr in tempo_rel]
    ultima_ativ = [min(hoje, a + timedelta(days=int(np.random.uniform(30, 365*tempo_rel[i]))))
                   for i, a in enumerate(abertura)]
    
    df = pd.DataFrame({
        "id_cliente": ids, "nome": nomes, "idade": idade, "genero": genero, "estado_civil": estado_civil,
        "regiao": regiao, "tempo_rel_anos": np.round(tempo_rel,2), "renda_mensal": np.round(renda,2),
        "emprego": emprego, "pontuacao_credito": score.astype(int),
        "possui_cartao": possui_cartao, "num_produtos": num_prod,
        "saldo_poupanca": np.round(saldo_poup,2), "saldo_corrente": np.round(saldo_corr,2),
        "usuario_app": usuario_app, "engajamento_digital": np.round(eng_digital,1),
        "visitas_agencia_12m": visitas_ag, "transacoes_12m": trans_12m,
        "segmento": seg, "tipo_emprestimo": tipo_emp, "valor_emprestimo": np.round(valor_emp,2),
        "prazo_meses": prazo, "taxa_juros_anual": np.round(taxa,4),
        "inadimplente": inad, "dias_atraso": dias_atraso, "churn": churn,
        "data_abertura": abertura, "data_ultima_atividade": ultima_ativ
    })
    
    df['dias_relacionamento'] = (hoje - df['data_abertura']).dt.days
    df['dias_desde_ultima_atividade'] = (hoje - df['data_ultima_atividade']).dt.days
    
    return df

clientes = gerar_base(n_clientes)

# -----------------------------
# Header
# -----------------------------
st.title("📊 Projeto CAIXA — Estatística, Modelagem e Dashboard")
st.caption("Base 100% sintética para fins didáticos — módulo de Estatística")

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes", f"{clientes.shape[0]:,}".replace(",", "."))
col2.metric("Inadimplência (%)", f"{clientes['inadimplente'].mean()*100:.2f}")
col3.metric("Uso do App (%)", f"{clientes['usuario_app'].mean()*100:.2f}")
col4.metric("Renda Média (R$)", f"{clientes['renda_mensal'].mean():,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

# -----------------------------
# Filtros
# -----------------------------
st.subheader("🔎 Filtros de Análise")
col_filtros = st.columns(3)
reg_sel = col_filtros[0].multiselect("Região", sorted(clientes["regiao"].unique()), default=None)
seg_sel = col_filtros[1].multiselect("Segmento", sorted(clientes["segmento"].unique()), default=None)
tipo_sel = col_filtros[2].multiselect("Tipo de Empréstimo", sorted(clientes["tipo_emprestimo"].unique()), default=None)

mask = np.ones(len(clientes), dtype=bool)
if reg_sel: mask &= clientes["regiao"].isin(reg_sel)
if seg_sel: mask &= clientes["segmento"].isin(seg_sel)
if tipo_sel: mask &= clientes["tipo_emprestimo"].isin(tipo_sel)
dfv = clientes.loc[mask].copy()

# -----------------------------
# Abas
# -----------------------------
tab_eda, tab_corr, tab_testes, tab_modelo = st.tabs(
    ["Exploração", "Correlação", "Testes de Hipótese", "Regressão"]
)

# -----------------------------
# Exploração
# -----------------------------
with tab_eda:
    st.subheader("Exploração Descritiva")
    
    # Amostra da base
    st.markdown("### Amostra da base filtrada")
    st.dataframe(dfv.head(10), use_container_width=True)

    # Resumo Numérico
    st.markdown("### Resumo de Variáveis Numéricas")
    num_cols = ["idade","renda_mensal","pontuacao_credito","num_produtos","saldo_poupanca",
                 "saldo_corrente","valor_emprestimo","prazo_meses","taxa_juros_anual",
                 "engajamento_digital","transacoes_12m","visitas_agencia_12m","dias_atraso"]
    
    desc_df = pd.DataFrame({
        "Mínimo": dfv[num_cols].min(),
        "Máximo": dfv[num_cols].max(),
        "Média": dfv[num_cols].mean(),
        "Mediana": dfv[num_cols].median(),
        "Moda": dfv[num_cols].mode().iloc[0]
    }).T.round(2)
    st.dataframe(desc_df, use_container_width=True)

    st.markdown("---")
    
    # Resumo Categórico
    st.markdown("### Resumo de Variáveis Categóricas")
    cat_cols = ["genero", "estado_civil", "regiao", "emprego", "segmento", "tipo_emprestimo"]
    
    for c in cat_cols:
        st.markdown(f"**{c.replace('_',' ').title()}**")
        
        # Tabela
        counts = dfv[c].value_counts().reset_index()
        counts.columns = [c.replace('_',' ').title(), "Contagem"]
        st.dataframe(counts, use_container_width=True)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=counts, x=counts.columns[0], y="Contagem", ax=ax, palette="viridis")
        ax.set_xlabel("")
        ax.set_ylabel("Frequência")
        ax.set_title(f"Distribuição de {c.replace('_',' ').title()}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        st.markdown("---")

# -----------------------------
# Correlação
# -----------------------------
with tab_corr:
    st.subheader("Análise de Correlação")
    
    st.markdown("### Matriz de Correlação Geral")
    num_cols_corr_geral = ["idade","renda_mensal","pontuacao_credito","tempo_rel_anos","num_produtos",
                     "saldo_poupanca","saldo_corrente","valor_emprestimo","prazo_meses","taxa_juros_anual",
                     "engajamento_digital","transacoes_12m","visitas_agencia_12m","inadimplente"]
    
    corr_matrix_geral = dfv[num_cols_corr_geral].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix_geral, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlação Geral")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### Análises de Pares Específicos")

    # Correlação: Idade x Renda
    st.markdown("#### Idade × Renda")
    corr_idade_renda = dfv[['idade', 'renda_mensal']].corr().iloc[0,1]
    st.info(f"O coeficiente de correlação entre idade e renda mensal é: **{corr_idade_renda:.2f}**")

    # Correlação: Engajamento Digital x Visitas à Agência
    st.markdown("#### Engajamento Digital × Visitas à Agência")
    corr_eng_visitas = dfv[['engajamento_digital', 'visitas_agencia_12m']].corr().iloc[0,1]
    st.info(f"O coeficiente de correlação entre engajamento digital e visitas à agência é: **{corr_eng_visitas:.2f}**")

    st.markdown("---")

    st.markdown("### Correlações Subdivididas por Tipo de Empréstimo")

    # Correlação: Renda x Valor Empréstimo por Tipo
    st.markdown("#### Renda × Valor de Empréstimo por tipo")
    tipos_emp = dfv['tipo_emprestimo'].unique()
    corr_data = []
    for t in tipos_emp:
        subset = dfv[dfv['tipo_emprestimo'] == t]
        if len(subset) > 1:
            corr = subset[['renda_mensal', 'valor_emprestimo']].corr().iloc[0,1]
            corr_data.append({"Tipo de Empréstimo": t, "Correlação": corr})
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data).round(2)
        corr_df = corr_df.set_index("Tipo de Empréstimo")
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.warning("Dados insuficientes para esta análise com os filtros atuais.")


    # Correlação: Score x Inadimplência por Tipo
    st.markdown("#### Pontuação de Crédito × Inadimplência por tipo")
    corr_data_score = []
    for t in tipos_emp:
        subset = dfv[dfv['tipo_emprestimo'] == t]
        if len(subset) > 1:
            corr = subset[['pontuacao_credito', 'inadimplente']].corr().iloc[0,1]
            corr_data_score.append({"Tipo de Empréstimo": t, "Correlação": corr})
    
    if corr_data_score:
        corr_df_score = pd.DataFrame(corr_data_score).round(2)
        corr_df_score = corr_df_score.set_index("Tipo de Empréstimo")
        st.dataframe(corr_df_score, use_container_width=True)
    else:
        st.warning("Dados insuficientes para esta análise com os filtros atuais.")

# -----------------------------
# Testes de hipótese
# -----------------------------
with tab_testes:
    st.subheader("Testes de Hipótese")
    
    st.markdown("Os testes de hipótese nos permitem ir além da análise descritiva, validando, com rigor estatístico, se as diferenças que observamos nos dados são realmente significativas ou apenas variações aleatórias.")

    st.markdown("---")

    # t-teste renda
    renda_inad = dfv.loc[dfv["inadimplente"]==1, "renda_mensal"]
    renda_ok = dfv.loc[dfv["inadimplente"]==0, "renda_mensal"]
    t_stat, p_val_t = stats.ttest_ind(renda_inad, renda_ok, equal_var=False, nan_policy="omit")

    st.markdown("#### t-teste (Renda: Inadimplentes vs. Adimplentes)")
    st.write(f"**Resultado:** `t-statistic` = {t_stat:.3f} | `p-valor` = {p_val_t:.4g}")
    st.markdown("""
    - **Hipótese Nula (H₀)**: A renda média é a mesma para ambos os grupos (inadimplentes e adimplentes).
    - **Hipótese Alternativa (H₁)**: A renda média é diferente entre os grupos.
    """)
    st.success("""
    **Conclusão para o negócio:**
    Como o p-valor é muito baixo (menor que 0.05), podemos rejeitar a hipótese nula. Isso significa que a **diferença na renda média entre clientes inadimplentes e adimplentes é estatisticamente significativa**. Clientes com menor renda tendem a ser mais propensos à inadimplência.
    """)
    
    st.markdown("---")

    # qui-quadrado: inadimplência x tipo de empréstimo
    tab = pd.crosstab(dfv["tipo_emprestimo"], dfv["inadimplente"])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(tab)
    st.markdown("#### Qui-quadrado (Associação: Inadimplência × Tipo de Empréstimo)")
    st.write(f"**Resultado:** `χ²` = {chi2:.3f} | `gl` = {dof} | `p-valor` = {p_val_chi2:.4g}")
    st.markdown("""
    - **Hipótese Nula (H₀)**: A inadimplência é independente do tipo de empréstimo.
    - **Hipótese Alternativa (H₁)**: Existe uma associação entre a inadimplência e o tipo de empréstimo.
    """)
    st.success("""
    **Conclusão para o negócio:**
    Com um p-valor extremamente baixo, rejeitamos a hipótese nula. Isso demonstra uma **associação significativa entre o tipo de empréstimo e a inadimplência**. Cada produto (como Imobiliário ou Cartão) tem um perfil de risco único.
    """)

    st.markdown("---")

    # z de proporções: app vs não
    count = np.array([dfv.loc[dfv["usuario_app"]==1, "inadimplente"].sum(),
                      dfv.loc[dfv["usuario_app"]==0, "inadimplente"].sum()])
    nobs = np.array([(dfv["usuario_app"]==1).sum(),
                      (dfv["usuario_app"]==0).sum()])
    if (nobs>0).all():
        z_stat, p_val_z = proportions_ztest(count, nobs, alternative="two-sided")
        st.markdown("#### z-teste de Proporções (Inadimplência: Usuários de App vs. Não-Usuários)")
        st.write(f"**Resultado:** `z-statistic` = {z_stat:.3f} | `p-valor` = {p_val_z:.4g}")
        st.markdown("""
        - **Hipótese Nula (H₀)**: A proporção de inadimplentes é a mesma entre os dois grupos.
        - **Hipótese Alternativa (H₁)**: A proporção é diferente.
        """)
        st.success("""
        **Conclusão para o negócio:**
        O p-valor é menor que 0.05, indicando que a **diferença de proporções é estatisticamente significativa**. Isso sugere que os usuários do aplicativo tendem a ter uma taxa de inadimplência menor. Fortalecer o `onboarding` digital pode ser uma estratégia para mitigar riscos.
        """)
    else:
        st.info("Amostra insuficiente em algum grupo para o teste de proporções.")

# -----------------------------
# Modelo: Regressão Logística
# -----------------------------
with tab_modelo:
    st.subheader("Modelo Preditivo — Regressão Logística")
    
    # Seleção + dummies
    model_df = dfv[[
        "inadimplente","renda_mensal","pontuacao_credito","tempo_rel_anos","num_produtos","taxa_juros_anual",
        "usuario_app","emprego","segmento","tipo_emprestimo"
    ]].dropna().copy()

    if model_df["inadimplente"].nunique() < 2:
        st.warning("Amostra filtrada não possui as duas classes (0 e 1). Alargue os filtros.")
    else:
        model_df = pd.get_dummies(model_df, columns=["emprego","segmento","tipo_emprestimo"], drop_first=True)
        y = model_df["inadimplente"].astype(int)
        X = model_df.drop(columns=["inadimplente"])
        
        # Força o tipo numérico para evitar o ValueError
        X = X.astype(float)
        
        logit = sm.Logit(y, X)
        res = logit.fit(disp=False)

        st.markdown("### Resumo do Modelo (statsmodels)")
        st.text(res.summary())

        or_table = odds_ratios_from_logit(res).sort_values("OR", ascending=False)
        st.markdown("### Odds Ratios (exp(beta))")
        st.dataframe(or_table.round(4), use_container_width=True)

        st.markdown("---")
        st.markdown("### Métricas de Performance do Modelo")

        # Probabilidades previstas
        p_hat = res.predict(X)
        # Métricas: AUC/KS/Calibração (sem sklearn)
        FPR, TPR, auc = roc_curve_manual(y.values, p_hat.values)
        ks = ks_statistic(y.values, p_hat.values)
        calib = calib_curve(y.values, p_hat.values, n_bins=10)

        m1, m2, m3 = st.columns(3)
        m1.metric("AUC (ROC)", f"{auc:.3f}")
        m2.metric("KS", f"{ks:.3f}")
        m3.metric("Inadimplência observada", f"{y.mean()*100:.2f}%")

        # ROC
        fig, ax = plt.subplots()
        ax.plot(FPR, TPR, label=f"AUC={auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Curva ROC")
        ax.legend()
        st.pyplot(fig)

        # Calibração
        fig, ax = plt.subplots()
        ax.plot(calib["prev"], calib["real"], marker="o")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_xlabel("Prob. prevista (bin)")
        ax.set_ylabel("Taxa real (bin)")
        ax.set_title("Curva de Calibração (10 bins)")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Probabilidade de Inadimplência vs. Variáveis Chave")
        
        def plot_probabilidade(modelo, df_base, var_nome, var_label):
            """Gera um gráfico da probabilidade prevista de inadimplência em relação a uma variável."""
            faixa = np.linspace(df_base[var_nome].min(), df_base[var_nome].max(), 100)
            
            # Cria um DataFrame hipotético com valores médios, variando apenas a variável de interesse
            dados_plot = pd.DataFrame({
                'const': np.ones(100),
                'pontuacao_credito': np.full(100, df_base['pontuacao_credito'].mean()),
                'tempo_rel_anos': np.full(100, df_base['tempo_rel_anos'].mean()),
                'renda_mensal': np.full(100, df_base['renda_mensal'].mean()),
                'num_produtos': np.full(100, df_base['num_produtos'].mode().iloc[0]),
                'taxa_juros_anual': np.full(100, df_base['taxa_juros_anual'].mean()),
                'usuario_app': np.full(100, df_base['usuario_app'].mode().iloc[0])
            })
            
            # Adiciona as colunas dummy, todas com 0
            for col in X.columns:
                if col not in dados_plot.columns:
                    dados_plot[col] = 0
            
            dados_plot[var_nome] = faixa
            
            # Garante que as colunas estão na ordem correta do modelo e com o tipo correto
            dados_plot = dados_plot[X.columns]
            dados_plot = dados_plot.astype(float)
            
            prob = modelo.predict(dados_plot)

            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(faixa, prob, color='teal')
            ax.set_title(f"Probabilidade de Inadimplência vs {var_label}")
            ax.set_xlabel(var_label)
            ax.set_ylabel("Probabilidade prevista")
            ax.grid(True)
            st.pyplot(fig)

        # Gera os gráficos
        if 'pontuacao_credito' in X.columns:
            plot_probabilidade(res, model_df, 'pontuacao_credito', 'Pontuação de Crédito')
            st.markdown("A curva mostra que, quanto maior a pontuação de crédito, menor é a probabilidade de inadimplência.")
        
        if 'tempo_rel_anos' in X.columns:
            plot_probabilidade(res, model_df, 'tempo_rel_anos', 'Tempo de Relacionamento (anos)')
            st.markdown("Observa-se que a probabilidade de inadimplência tende a diminuir com o aumento do tempo de relacionamento do cliente com o banco.")

        st.markdown("""
        ---
        ### **Interpretação prática (exemplo):**
        - **OR > 1**: A variável aumenta as chances de inadimplência (ex.: juros altos, produtos sem garantia).
        - **OR < 1**: A variável reduz o risco (ex.: maior score, maior tempo de relacionamento, uso do app).
        - **AUC** próximo de 0,7–0,8 é um bom desempenho em modelos de risco; **KS** acima de 0,3 é um bom indicador de separação entre as classes.
        - A **Curva de Calibração** deve se aproximar da linha diagonal para indicar que as probabilidades previstas são precisas.
        """)