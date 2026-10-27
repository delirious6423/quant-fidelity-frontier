---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0.txt
library_name: gguf
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- gguf
- gemma
- vision
- optima-tools
---

# Optima Tools — Chat et raisonnement — Medium

Paquet **Medium** du chat d'Optima Tools (identifiant `chat.medium`) : **Gemma 4 26B-A4B** (instruction-tuned), quantification **Q5_K_M**, avec son projecteur multimodal **BF16** pour l'analyse d'images.

## Statut

- Les deux fichiers sont ceux **configurés dans Optima Tools** sur le poste de développement. Leurs SHA-256 ont été **recalculés localement** et sont identiques à ceux des métadonnées LFS de Hugging Face.
- Le GGUF Q5_K_M est celui de la révision `cb1f989321df6baa8c647aea074fbad465962bb1` du dépôt de bartowski (3 mai 2026). **Il diffère du GGUF de la révision la plus récente** (`10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d`, 19 319 198 848 octets, chat template mis à jour) : Optima Tools conserve la version qu'il utilise.
- Ce contrôle est une **vérification d'intégrité, pas un test d'exécution** : aucun test n'a été réalisé lors de cette préparation, et aucune performance ni compatibilité matérielle n'est annoncée.
- Le téléchargement depuis l'application n'est pas encore ouvert.

## Fichiers du modèle

| Fichier | Rôle | Taille | SHA-256 |
|---|---|---:|---|
| `google_gemma-4-26B-A4B-it-Q5_K_M.gguf` | Modèle de langage (Gemma 4 26B-A4B, Q5_K_M) | 19,32 Go | `948330216e3cf5aab7c995ffcb9530b1a76dec35a4ce1134db9dd0c1fab96e6b` |
| `mmproj-google_gemma-4-26B-A4B-it-bf16.gguf` | Projecteur multimodal BF16 : nécessaire à l'analyse des images | 1,19 Go | `b3ee6c97d5a5bb1ae9eb93bf14c1d1b51a0179a45ac1076b195931814c759e1e` |

Les deux fichiers se placent dans le même dossier : le moteur (llama.cpp) retrouve le projecteur à côté du modèle.

## Source

- Dépôt : https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF (conversion GGUF par bartowski ; modèle d'origine `google/gemma-4-26B-A4B-it`, Google DeepMind)
- Q5_K_M : révision `cb1f989321df6baa8c647aea074fbad465962bb1` ; projecteur et `README-source.md` : révision de référence `10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d`.
- Transfert : duplication des fichiers LFS côté serveur Hugging Face, puis commit. Aucun fichier modifié.

## Licence et crédits

Apache License 2.0 : texte intégral dans `LICENSE`. Google DeepMind (modèle d'origine), bartowski (conversion GGUF), Optima Tools (préparateur et distributeur), sources et modifications : `CREDITS.md` et `CREDITS.json`.

## Documents du dépôt

| Fichier | Contenu |
|---|---|
| `README-source.md` | README officiel de la révision de référence, copie identique (il décrit la révision la plus récente du dépôt source) |
| `LICENSE` | Licence Apache 2.0 |
| `CREDITS.md`, `CREDITS.json` | Crédits, licence, modifications |
| `SOURCE.json` | Provenance exacte de chaque fichier, révisions, tailles, SHA-256 |
| `.optima-package.json` | Manifeste lu par Optima Tools (taille et SHA-256 de chaque fichier du modèle) |

## Droits des auteurs d'origine

Les auteurs d'origine conservent l'ensemble de leurs droits. Optima Tools n'est pas le créateur de ce modèle : ce dépôt en prépare la distribution vérifiée pour l'application, avec mention des sources, des licences et des crédits.
