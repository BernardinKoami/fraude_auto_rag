# FraudScan — Détection de fraude assurance auto par RAG & LLM
**IA & Assurance**, ISFA 2025–2026  
Auteurs : **Apélété Primos ATISSO · Massile BADARO · Koami Bernardin ZANTE**

---

## Problématique métier

La fraude à l'assurance automobile représente entre **2,5 et 3 milliards d'euros** de pertes annuelles pour le secteur français (FFA, 2023), soit 5 à 7 % des indemnisations versées. Ce projet implémente un pipeline de détection automatisé combinant :

- **Modèle relationnel** : 7 bases de référence interdépendantes (assurés, contrats, garages, experts, AGIRA externe, blacklist) alimentant 16 règles de suspicion dont 5 nouvelles (hors garantie, AGIRA ext, blacklist, non-SRA, expert non certifié)
- **RAG sémantique** : recherche des sinistres historiques les plus similaires via embeddings vectoriels (`all-MiniLM-L6-v2`, 384 dim, ChromaDB HNSW)
- **Score réseau** : malus binaires issus des bases de référence, plafonné à 0,50
- **Agent LLM Expert** : analyse métier structurée générée par LLaMA 3.3 70B (Groq, gratuit)
- **Juge LLM** : validation automatique de la qualité des analyses avec boucle de correction (max 2 tentatives)

**Résultats clés :** AUC-ROC = 0,7748 · Précision@20 = 100 % · Économie coût = 89 % par calibrage du seuil · Coût LLM = 0,00 $

---

## Architecture du pipeline

```
Sinistre entrant
      │
      ▼
[Bases de référence] 7 tables relationnelles — lookups O(1)
      │
      ▼
[Module 1] Génération 10 000 sinistres — 16 règles R1–R16
      │
      ▼
[Module 2] EDA + analyse réseau + analyse de saisonnalité
      │
      ▼
[Module 3-4] RAG : claim_to_document() → embeddings all-MiniLM-L6-v2
             → ChromaDB HNSW (8 000 vecteurs train)
      │
      ▼
[Module 5] Score hybride : 0,70 × score_RAG + 0,30 × score_réseau + 0,10 bonus
      │                     ├─ FAIBLE  < 0,40   → traitement standard
      │                     ├─ MODÉRÉ  0,40–0,70 → pièces complémentaires
      │                     └─ ÉLEVÉ   ≥ 0,70   → gel + enquête terrain
      ▼
[Module 6] Agent Expert LLM (LLaMA 3.3 70B) → JSON structuré
      │        score_fraude · niveau_risque · motifs · recommandation
      ▼
[Module 6] Juge LLM → évaluation 5 critères (≥ 4,0/5 = VALIDÉ)
      │         └─ boucle de correction (max 2 retries si À REVOIR)
      ▼
[Module 7-10] Évaluation : AUC-ROC, F1, Précision@k + calibrage seuil coût
      │
      ▼
[Module 9] Graphe réseau NetworkX — clusters de fraude organisée
      │
      ▼
[Module 12] API FastAPI — /score · /analyze · /analyze_with_judge
      │
      ▼
[Module 13] Interface FraudScan (fraudscan.html) — démo autonome navigateur
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` (384 dim, CPU local) |
| Vector store | `chromadb` PersistentClient · HNSW · distance cosine · k=5 |
| LLM | `llama-3.3-70b-versatile` via Groq (SDK OpenAI, `base_url` redirigé) |
| Graphe réseau | `networkx` — clustering Louvain, centralité |
| Data | `pandas`, `numpy`, `faker` (locale fr_FR) |
| Évaluation | `scikit-learn` (AUC-ROC, F1, Précision@k, calibrage coût) |
| API | `FastAPI` + `uvicorn` |
| Interface | HTML/JS autonome (scoring JS + API Groq directe) |
| Python | 3.11+ |

---

## Structure du dépôt

```
.
├── Notebook_Final_v2.ipynb      # Notebook principal (13 modules, 94 cellules)
├── app.py                       # API REST FastAPI (4 endpoints)
├── pipeline.py                  # Pipeline production : RAG, scoring, agents LLM
├── fraudscan.html               # Interface démo interactive autonome
├── requirements.txt
├── README.md
├── skills/
│   ├── expert_fraude_production.md   # Prompt système Agent Expert (16 signaux)
│   └── juge_production.md            # Prompt système Juge LLM (5 critères)
├── data/
│   ├── claims_10k.csv           # Dataset principal (10 000 sinistres, 35 colonnes)
│   ├── ref_assures.csv          # 6 000 assurés
│   ├── ref_contrats.csv         # 4 000 contrats (matrice couverture 5×15)
│   ├── ref_garages.csv          # 500 garages (accréditation SRA)
│   ├── ref_experts.csv          # 200 experts (certification tribunal)
│   ├── ref_agira_externe.csv    # 3 453 sinistres cross-compagnies
│   ├── ref_blacklist.csv        # 50 assurés blacklistés FFA/ALFA
│   └── ref_contrat_insure.csv   # 5 447 liens contrat–assuré (many-to-many)
├── artifacts/
│   ├── eda_production.png       # Visualisations EDA (6 graphiques)
│   ├── saisonnalite_prod.png    # Analyse temporelle (4 graphiques)
│   ├── performances_prod.png    # Courbes ROC + Précision-Rappel
│   ├── confusion_prod.png       # Matrice de confusion
│   ├── calibrage_prod.png       # Courbe de calibrage du seuil par coût
│   ├── metrics_prod.json        # Métriques finales (AUC, F1, Précision@k...)
│   └── predictions_test_prod.csv
└── network/
    ├── garage_stats.csv         # Taux de fraude par garage (9 suspects confirmés)
    ├── expert_stats.csv         # Taux de fraude par expert
    └── fraud_network.gexf       # Graphe réseau (6 clusters, exportable Gephi)

# Générés automatiquement (non versionnés) :
# data/claims_10k.json           ← Module 1
# data/documents_10k.jsonl       ← Module 3
# vector_store/                  ← Module 3-4 (ChromaDB, ~12 MB)
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://forge.univ-lyon1.fr/p2408012/fraude_auto_rag.git
cd fraude_auto_rag
```

### 2. Créer l'environnement Python 3.11

```bash
conda create -n fraud_env python=3.11
conda activate fraud_env
```

ou avec venv :

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé Groq

Clé **gratuite** disponible sur [console.groq.com](https://console.groq.com).

```bash
export GROQ_API_KEY="gsk_..."
```

Ou créer un fichier `.env` (jamais commité) :

```bash
echo "GROQ_API_KEY=gsk_..." > .env
```

---

## Exécution du notebook

```bash
jupyter notebook Notebook_Final_v2.ipynb
```

**Ordre d'exécution recommandé :**

| Module | Contenu | Fichiers générés |
|---|---|---|
| 0 | Setup, imports, configuration Groq | — |
| 1 | Génération des 7 bases + 10 000 sinistres | `data/claims_10k.csv`, `data/ref_*.csv` |
| 2 | EDA, analyse réseau, **saisonnalité** | `artifacts/eda_production.png`, `artifacts/saisonnalite_prod.png` |
| 3 | Documents RAG | `data/documents_10k.jsonl` |
| 4 | Embeddings + ChromaDB (8 000 vecteurs) | `vector_store/` |
| 5 | Scoring hybride RAG + réseau | — |
| 6 | Agent Expert LLM + Juge + boucle correction | — |
| 7–10 | Évaluation, métriques, calibrage seuil | `artifacts/performances_prod.png`, `artifacts/metrics_prod.json` |
| 9 | Graphe réseau NetworkX | `network/fraud_network.gexf` |
| 11 | Limites & perspectives | — |
| 12 | Génération `app.py` FastAPI | `app.py` |
| 13 | Interface FraudScan inline | `fraudscan.html` |

> **Raccourci :** si `data/claims_10k.csv` et `data/ref_*.csv` sont déjà présents,
> démarrer directement au **Module 3** pour reconstruire uniquement l'index vectoriel.

> **ChromaDB :** en cas d'erreur `InternalError: readonly database` à la ré-exécution,
> le Module 4 supprime automatiquement l'ancien index avec `shutil.rmtree` avant de le recréer.

---

## Lancer l'API

```bash
export GROQ_API_KEY="gsk_..."
uvicorn app:app --port 8000
```

> **Note :** ne pas utiliser `--reload` — il surveille le dossier et provoque une boucle de redémarrage.

Documentation Swagger interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Endpoint | Description | Latence |
|---|---|---|
| `GET /health` | Statut du service + nombre de vecteurs indexés | < 50 ms |
| `POST /score` | Score hybride RAG + réseau, sans LLM | ~200 ms |
| `POST /analyze` | Score + analyse Agent Expert LLM | ~2–4 s |
| `POST /analyze_with_judge` | Score + Expert + Juge + boucle correction | ~5–8 s |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_000001",
    "insured_id": "INS_012345",
    "incident_type": "Collision simple",
    "incident_severity": "important",
    "incident_date": "2024-08-15",
    "incident_hour": 2,
    "incident_day_of_week": 6,
    "declaration_delay_days": 15,
    "region": "Île-de-France",
    "vehicle_brand": "BMW",
    "vehicle_age": 2,
    "total_claim_amount": 19800.0,
    "driver_age": 24,
    "months_as_customer": 3,
    "annual_income": 14000.0,
    "profession": "sans emploi",
    "witnesses": 0,
    "prior_claims": 3,
    "bodily_injuries": 0,
    "police_report": "Non",
    "garage_id": "GAR_0001",
    "expert_id": "AUCUN",
    "claims_last_36m_agira": 4,
    "incident_description": "Collision sur autoroute de nuit."
  }'
```

---

## Interface FraudScan (Module 13)

`fraudscan.html` est une interface web interactive **entièrement autonome** — aucun serveur requis.

Elle embarque directement dans le navigateur :
- Le scoring hybride kNN + score réseau (JavaScript)
- L'appel à l'API Groq pour l'analyse LLM (LLaMA 3.3 70B)
- Un corpus de 16 sinistres réels du dataset pour le kNN
- 3 exemples préchargés (FAIBLE · MODÉRÉ · PROD_DEMO_001)

### Utilisation

**Option 1 — Depuis le notebook**

```python
# Cellule 94 du notebook — affiche l'interface inline dans Jupyter
# La clé Groq est injectée automatiquement depuis GROQ_API_KEY
display_fraudscan()
```

**Option 2 — Standalone **

1. Ouvrir `fraudscan.html` dans Chrome
2. Entrer la clé Groq `gsk_...` dans le champ en haut de la page
3. Remplir le formulaire ou charger un exemple → cliquer **Analyser le sinistre**

> La clé Groq est transmise uniquement à `api.groq.com` — elle n'est jamais envoyée ailleurs.

### Fonctionnement interne

| Étape | Traitement | Résultat affiché |
|---|---|---|
| 1 | Scoring JS (kNN cosinus + score réseau) | Jauge score final, niveau, signaux actifs, voisins RAG |
| 2 | Appel `api.groq.com` (LLaMA 3.3 70B) | Analyse Expert : motifs, signaux réseau, recommandation |

---

## Résultats clés

| Métrique | Valeur (k=5, jeu de test 2 000 sinistres) |
|---|---|
| AUC-ROC | 0,7748 |
| Average Precision | 0,5864 (2,8× la baseline aléatoire) |
| F1-score (seuil optimal 0,281) | 0,6048 |
| **Précision@20** | **1,000 (100 %)** |
| Précision@100 | 0,720 |
| Coût seuil 0,5 | 2 280 000 € |
| **Coût seuil optimal 0,010** | **257 400 € (−89 %)** |
| Coût LLM total | 0,00 $ (Groq gratuit) |

**Sensibilité à k :**

| k | AUC-ROC | Statut |
|---|---|---|
| k=1 | 0,8796 | Instable |
| k=3 | 0,7985 | |
| **k=5** | **0,7748** | **Production** |
| k=10 | 0,7203 | Sur-lissage |

---

## Reproductibilité

- Toutes les données sont générées avec `random.seed(42)`, `numpy.random.seed(42)`, `Faker.seed(42)`
- Split stratifié avec `random_state=42`
- Les embeddings sont calculés localement (CPU, aucune dépendance externe)
- Le seul élément non déterministe est la sortie du LLM (`temperature=0.1`)
- Appeler `afficher_bilan()` après le Module 6 pour le résumé des tokens et coûts

---

## Limites identifiées

- **Données synthétiques** : taux de fraude ~30 % volontairement élevé (marché réel : 6–8 %) — métriques optimistes par construction
- **Embedding généraliste** : `all-MiniLM-L6-v2` non spécialisé IARD → CamemBERT-Assurance serait plus précis
- **Scalabilité ChromaDB** : ~500 000 vecteurs max en local → migration PostgreSQL + pgvector pour la production
- **AGIRA simplifiée** : l'accès réel nécessite une accréditation FFA et une API sécurisée
- **Hallucinations LLM** : la boucle Expert + Juge réduit le risque mais ne l'élimine pas — validation humaine obligatoire
- **AI Act & RGPD** : toute décision automatisée impactant un assuré requiert un audit d'équité et un droit à la contestation



