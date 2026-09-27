# Changelog

## [0.3.0] — 2026-09-27

### Ajouté
- Adaptation automatique légère des hyperparamètres du processus gaussien (`adapt_hyperparams`)
- Échantillonnage quasi-Monte-Carlo (Sobol) pour les candidats et la phase initiale
- Paramètre `xi` pour contrôler l’exploration de l’Expected Improvement
- Propriété `n_observations`

### Amélioré
- Stabilité numérique : décomposition de Cholesky à la place de l’inversion directe
- Meilleure couverture de l’espace de recherche grâce à Sobol
- Documentation et exemples enrichis

### Technique
- Version passée à 0.3.0
- README refondu (impact énergétique, économie de données, cas d’usage)

## [0.2.0] — version précédente

- API unifiée 1D (scalaire) / nD (vectorielle)
- Processus gaussien + Expected Improvement
- Estimation du meilleur point via la moyenne du modèle (correction du biais optimiste)
