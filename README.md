# Instagram Network Analysis

Um MVP para análise de redes de seguidores no Instagram, permitindo identificar influenciadores e centros de interesse comuns entre os seguidores de um perfil-alvo.

## Visão Geral

Este projeto implementa uma ferramenta completa para analisar redes de seguidores no Instagram. Ele coleta seguidores de um perfil-alvo, identifica quem eles seguem, e agrega esses dados para gerar um ranking das contas mais seguidas por esse grupo, revelando influenciadores e centros de interesse comuns.

### Principais Funcionalidades

- **Coleta de Seguidores**: Coleta assíncrona de seguidores de um perfil-alvo
- **Coleta de Seguidos**: Coleta paralela de quem os seguidores seguem
- **Processamento de Dados**: Agregação e análise para identificar contas influentes
- **Visualização**: Gráficos e dashboards para insights visuais
- **Exportação**: Rankings em múltiplos formatos (CSV, JSON)
- **Integração com GCS**: Opção para salvar dados no Google Cloud Storage

## Estrutura do Projeto

```
instagram-network-analysis/
│
├── config/                  # Configurações centralizadas
│   ├── settings.py          # Parâmetros, limites, credenciais
│   └── logging_config.py    # Configuração de logs
│
├── data/                    # Estrutura de dados organizada
│   ├── raw/                 # Dados brutos (followers/followings)
│   ├── processed/           # Dados processados e checkpoints
│   └── results/             # Rankings e visualizações
│
├── notebooks/               # Notebooks para execução e análise
│   ├── pipeline.ipynb       # Pipeline principal para Google Colab
│   └── analysis.ipynb       # Análises adicionais
│
├── src/                     # Código modularizado
│   ├── collectors/          # Módulos de coleta de dados
│   ├── processors/          # Análise e processamento
│   ├── utils/               # Utilitários (auth, rate limiting, etc.)
│   └── visualizers/         # Geradores de visualizações
│
├── tests/                   # Testes automatizados
└── docs/                    # Documentação
```

## Requisitos

- Python 3.7+
- Google Colab Pro (recomendado para execução)
- Pacotes Python: requests, pandas, numpy, matplotlib, seaborn, networkx, aiohttp, asyncio, scikit-learn

## Instalação e Configuração

### Opção 1: Google Colab (Recomendado)

1. Faça upload dos arquivos do projeto para o Google Drive
2. Abra o notebook `notebooks/pipeline.ipynb` no Google Colab
3. Siga as instruções no notebook para configurar e executar a análise

### Opção 2: Instalação Local

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/instagram-network-analysis.git
   cd instagram-network-analysis
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Execute o notebook principal:
   ```
   jupyter notebook notebooks/pipeline.ipynb
   ```

## Uso

### Autenticação no Instagram

O sistema utiliza cookies de sessão para autenticação, evitando problemas com 2FA:

1. Faça login no Instagram pelo navegador
2. Abra as ferramentas de desenvolvedor (F12)
3. Vá para a aba "Application" ou "Storage"
4. Em "Cookies", encontre "instagram.com"
5. Copie o valor do cookie "sessionid"
6. Cole este valor quando solicitado no notebook

### Execução da Análise

1. Abra o notebook `notebooks/pipeline.ipynb`
2. Insira o nome de usuário do perfil-alvo para análise
3. Configure os parâmetros de coleta (número máximo de seguidores, etc.)
4. Execute todas as células do notebook
5. Os resultados serão salvos na pasta `data/results/`

### Análises Avançadas

Para análises mais detalhadas:

1. Execute primeiro o pipeline principal completo
2. Abra o notebook `notebooks/analysis.ipynb`
3. Execute as células para gerar visualizações e insights adicionais

## Parâmetros de Configuração

Os principais parâmetros podem ser ajustados em `config/settings.py`:

### Coleta de Dados
- `MAX_FOLLOWERS_PER_RUN`: Número máximo de seguidores a coletar (padrão: 20000)
- `MAX_FOLLOWING_PER_USER`: Número máximo de seguidos por usuário (padrão: 1000)
- `PARALLEL_REQUESTS`: Número de requisições paralelas (padrão: 3)

### Rate Limiting
- `MAX_REQUESTS_PER_HOUR`: Limite de requisições por hora (padrão: 150)
- `MIN_REQUEST_INTERVAL`: Intervalo mínimo entre requisições (padrão: 2.0s)

### Processamento
- `MIN_FOLLOWERS_THRESHOLD`: Limite mínimo de seguidores para inclusão na análise
- `OUTLIER_DETECTION`: Configurações para detecção de outliers

## Medidas de Segurança

O sistema implementa várias medidas para evitar bloqueios:

- Delays adaptativos baseados na resposta do servidor
- Detecção proativa de sinais de bloqueio
- Sistema de retentativas com backoff exponencial
- Checkpoints para retomada em caso de interrupção

## Resultados e Visualizações

A análise gera diversos resultados:

- Rankings das contas mais seguidas pelos seguidores do perfil-alvo
- Métricas de influência e penetração para cada conta
- Visualizações de rede mostrando relações entre contas
- Identificação de clusters de interesse
- Dashboard completo com múltiplas visualizações

## Integração com Google Cloud Storage

Para salvar dados no GCS:

1. Configure as credenciais do GCS em `config/settings.py`
2. Ative a opção `USE_GCS` no notebook principal
3. Os dados serão salvos tanto localmente quanto no bucket GCS configurado

## Limitações Conhecidas

- O Instagram pode limitar o número de requisições, mesmo com as medidas implementadas
- Contas privadas não permitem acesso à lista de seguidos
- A API não oficial do Instagram pode mudar, exigindo atualizações no código

## Contribuições

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um novo Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contato

Para questões ou suporte, entre em contato através de [seu-email@exemplo.com].
