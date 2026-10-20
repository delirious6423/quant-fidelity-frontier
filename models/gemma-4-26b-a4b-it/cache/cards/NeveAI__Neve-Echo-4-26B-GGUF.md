---
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
  - NeveAI
  - Neve
  - Echo
---


<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/68a3ba234a7dfca33d72eee2/uEgq1cKh6eWYsyWKbcqch.png" width="50%">
</div>

<h1 align="center">

Neve-Echo-4-26B-GGUF</h1>

<div align="center">
  <a href="https://github.com/NeveIA">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/68a3ba234a7dfca33d72eee2/KQa7-ajynUAhTS-kxNtYT.png" width="20%" alt="NeveAI GitHub">
  </a>
</div>

## Introdução

O **Neve Echo 4** é um modelo de linguagem de última geração focado em **uso geral e raciocínio para tarefas variadas**. Esta versão em formato GGUF foi otimizada pela NeveAI para oferecer o equilíbrio ideal entre precisão lógica e eficiência computacional.

---

## Destaques do Modelo

Este modelo foi desenvolvido para uso geral e execução de tarefas diversas, focando em:

*   **Raciocínio Avançado (Thinking):** Projetado como um modelo altamente capaz de raciocínio, com suporte a modos de pensamento estruturado para tarefas complexas.
*   **Uso Geral e Produtividade:** Otimizado para uma ampla variedade de tarefas como geração de texto, assistência, explicações, planejamento e automação.
*   **Multimodalidade (Texto + Imagem):** Capaz de processar entradas multimodais com suporte a diferentes resoluções e proporções.
*   **Tool Calling e System Prompt:** Suporte nativo a chamadas de função e ao papel `system`, permitindo maior controle e integração com ferramentas externas.

## Benchmark de Performance

O Neve Echo 4 demonstra desempenho alinhado a modelos de ponta em múltiplas categorias:

| Categoria | Benchmark | Neve Echo 4 | Gemma 4 26B A4B |
| :--- | :--- | :---: | :---: |
| **Knowledge** | MMLU Pro | **85.2%** | 82.6% |
| **Reasoning** | GPQA Diamond | **84.3%** | 82.3% |
| **Math** | AIME 2026 | **89.2%** | 88.3% |
| **General** | BigBench Extra Hard | **74.4%** | 64.8% |

---

## Detalhes da Arquitetura

- **Arquitetura:** Mixture of Experts (MoE) com atenção híbrida (local + global intercalado).
- **Parâmetros:** ~26B totais (com subset ativo por token para eficiência).
- **Janela de Contexto:** Até 256K tokens.
- **Camadas:** Arquitetura profunda com atenção global na camada final.
- **MoE:** Roteamento dinâmico de experts (subset ativo por inferência), otimizando performance e uso de memória.

## Como utilizar (GGUF)

Este modelo é compatível com `llama.cpp`, `Ollama`, `LM Studio` e outras ferramentas que suportam o formato GGUF. 
Foco direcionado ao uso do modelo na plataforma autoral da organização [NeveAI](https://github.com/Etamus/NeveAI)

## Licença

Este repositório e os pesos do modelo estão licenciados sob a [Licença Apache 2.0](LICENSE).

## Contato

Se tiver qualquer dúvida, por favor, levante um issue ou entre em contato conosco em [NeveIA](https://github.com/NeveIA).