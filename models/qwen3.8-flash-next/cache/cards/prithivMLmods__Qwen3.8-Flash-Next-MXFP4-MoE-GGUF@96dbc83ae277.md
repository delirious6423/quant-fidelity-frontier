---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
language:
- en
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen4_exp
- llama-cpp
- llama-cli
- mtp
- speculative-decoding
- qwen4_exp
- microscaling
- text-generation-inference
- moe
- Qwen3.8
- mxfp4_moe
- open-weight
- mmproj
- nvidia-blackwell
- rtx-pro-6000
- rtx-5090
- SM120
- GB202
- B200
- B300
- FP4
- blackwell
- kv-cache
- instruction-following
- multimodal
- MTP-head
---

# **Qwen3.8-Flash-Next-MXFP4-MoE-GGUF**

> **Qwen3.8-Flash-Next** is an open-weight multimodal Mixture-of-Experts (MoE) model and experimental architectural preview for Qwen4, combining 125B core parameters (6B activated per token across 10 routed and 1 shared expert out of 512) with 51B offload-friendly n-gram embedding parameters and a 4B multi-token prediction (MTP) head alongside a vision encoder. Across its 48 layers, it introduces a hybrid attention layout interleaving Gated DeltaNet with block-level **Qwen Sparse Attention (QSA)** to cut long-context latency, multi-branch **Gated Residual** streams for training stability, and a native 262,144-token context window extensible to 1M tokens via YaRN. Supporting text, image, and hour-scale video inputs with configurable `<think>` reasoning (`enable_thinking`, `preserve_thinking`, and `reasoning_effort`), it outperforms larger and proprietary peers across agentic coding, computer use, and multimodal benchmarks—including DeepSWE 1.1 (58.7), SWE-bench Pro (62.5), SWE-bench Multilingual (81.0), AndroidWorld (84.5), GPQA Diamond (91.7), LiveCodeBench v6 (91.9), and MathVision (95.7 with code interpreter)—and is released under the **Qwen Community 1.0** license for deployment via `Transformers`, `vLLM`, `SGLang`, and `TokenSpeed`.

> [!NOTE]
> MTP — Speculative Decoding: Yes

> [!NOTE]
> Multi-Token Prediction (MTP) is a speculative decoding technique that accelerates local inference by predicting multiple future tokens in a single forward pass. Unlike traditional speculative decoding, it does not require a separate, smaller "draft" model; the MTP prediction heads act as the drafter. In this repository, the MTP weights are shipped as a **separate GGUF file** **[Qwen3.8-Flash-Next-MTP.BF16.gguf](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next-MTP.BF16.gguf)** rather than embedded in the main model file. Load it alongside the main model GGUF to enable speculative decoding. The main model runs standalone without it.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| Qwen3.8-Flash-Next-MTP.BF16.gguf | BF16 (MTP) | 7.77 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next-MTP.BF16.gguf) | Standalone Multi-Token Prediction (MTP) file in BF16. Loaded alongside the main model to enable speculative decoding. Not a standalone model. |
| Qwen3.8-Flash-Next.MXFP4_MOE.gguf | MXFP4_MOE | 124 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next.MXFP4_MOE.gguf) | **MXFP4_MOE**: Highly optimized for **NVIDIA Blackwell** hardware accelerators **(e.g., RTX Pro 6000, RTX 50-series, B200, B300)**, offering native hardware-accelerated FP4 Tensor Core speeds. Main model file; pair with the MTP file for speculative decoding. |
| Qwen3.8-Flash-Next.mmproj-bf16.gguf | mmproj-bf16 | 908 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next.mmproj-bf16.gguf) | Multimodal projection file in BF16 format. Used for vision/image capabilities. |

## Run Inference (llama.cpp)

```bash
/path/to/llama.cpp/build/bin/llama-cli \
  -m /path/to/models/Qwen3.8-Flash-Next.MXFP4_MOE.gguf \
  -md /path/to/models/Qwen3.8-Flash-Next-MTP.BF16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl auto \
  -c 16384 \
  --cache-type-k q8_0 \
  --seed 42 \
  -n 16384 \
  -st \
  -p "Given a special linked list with n nodes where each node has two pointers a next pointer that points to the next node of the singly linked list, and a random pointer that points to the random node of the linked list."
```

## Output (llama.cpp)

<div style="
    background: rgba(255, 255, 0, 0.25); /* light yellow background */
    padding: 16px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 0, 0.5); /* light yellow border */
    margin: 16px 0;
">

<details>
  <summary>Expand the dropdown to view the full output.</summary>

> Given a special linked list with n nodes where each node has two pointers a next pointer that points to the next node of the singly linked list, and a random pointer that points to the random node of the linked list.

[Start thinking]

We need answer user. User: "Given a special linked list with n nodes where each node has two pointers a next pointer that points to the next node of the singly linked list, and a random pointer that points to the random node of the linked list." This sounds like prompt maybe wants clone? Need infer likely classic problem: copy linked list with random pointers. They might be asking algorithm? But not explicit question. Need respond perhaps explain problem and ask? Could provide solution: deep copy with O(1) space using interleaving or O(n) with hashmap. Need maybe mention constraints? Let's craft answer: if goal is to clone such list, provide approaches. Since user only statement, maybe expects completion? The phrase might be from interview question: "Given a special linked list... write a program to copy the linked list such that next and random pointers point to copies." Need provide detailed answer. We'll include problem: clone linked list with random pointers. Provide O(n) time, O(n) hash map and O(n) time O(1) extra space interleaving. Maybe also if just asking to detect/construct random pointers? Need answer general.

Need ensure final. Use analysis now. We'll formulate. Also maybe mention if random pointer can point to node outside? Usually within list. Need algorithm steps, code maybe Python/C++/Java? Could include one language or pseudocode. Since no language, use generic. Need be thorough. Maybe include definitions, example, edge cases.

Need final with headings. Let's produce.
[End thinking]

You are describing the classic **Copy a Linked List with Random Pointers** problem.

Each node has:

```text
node.next    → next node in the linked list
node.random  → random node in the linked list
```

The usual task is:

> Given the head of a linked list where every node has `next` and `random` pointers, return a deep copy of the list.  
> The copied list should contain new nodes, and the `next` and `random` pointers in the new list should point to copied nodes, not the original nodes.

---

## Example

Original list:

```text
1 → 2 → 3 → 4 → null

Random pointers:
1.random → 3
2.random → 1
3.random → 4
4.random → 2
```

Copied list:

```text
1' → 2' → 3' → 4' → null

1'.random → 3'
2'.random → 1'
3'.random → 4'
4'.random → 2'
```

All nodes are new, and no pointer from the copy should point to an original node.

---

# Approach 1: Using a Hash Map

## Idea

1. Create a new copy of every node.
2. Store a mapping:

```text
original node → copied node
```

3. Connect the copied list using:
- copied node’s `next` pointer
  - copied node’s `random` pointer

---

## Algorithm

```text
If head is null:
    return null

map = empty hash map

// Step 1: Create all nodes and store original → copy mapping
current = head
while current != null:
    copyNode = new Node(current.val)
    map[current] = copyNode
    current = current.next

// Step 2: Set next and random pointers of copied nodes
current = head
while current != null:
    copyNode = map[current]

    if current.next != null:
        copyNode.next = map[current.next]

    if current.random != null:
        copyNode.random = map[current.random]

    current = current.next

return map[head]
```

---

## Complexity

```text
Time Complexity:  O(n)
Space Complexity: O(n)
```

Because the hash map stores one entry per node.

---

# Approach 2: Using Interleaved Nodes — O(1) Extra Space

This is the optimal interview solution.

## Idea

Instead of using a hash map, temporarily weave each copied node directly after its original node.

If the original list is:

```text
1 → 2 → 3
```

Create:

```text
1 → 1' → 2 → 2' → 3 → 3'
```

Now each original node `current` has its copy as `current.next`.

This allows us to resolve random pointers easily:

```text
original random pointer points to current.random
so copied random pointer should point to current.random.next
```

Finally, separate the original list and copied list.

---

## Step-by-Step Process

### Step 1: Insert copy nodes after each original node

```text
Original:
1 → 2 → 3 → null

After insertion:
1 → 1' → 2 → 2' → 3 → 3' → null
```

---

### Step 2: Assign random pointers to copied nodes

For every original node `current`:

```text
If current.random != null:
    current.next.random = current.random.next
```

Why?

```text
current.next       → copy of current
current.random     → random original node
current.random.next → copy of that random original node
```

---

### Step 3: Separate original and copied lists

```text
1 → 1' → 2 → 2' → 3 → 3' → null

becomes:

Original:
1 → 2 → 3 → null

Copy:
1' → 2' → 3' → null
```

---

## Algorithm

```text
If head is null:
    return null

// Step 1: Create copy nodes and insert them after original nodes
current = head

while current != null:
    copyNode = new Node(current.val)

    copyNode.next = current.next
    current.next = copyNode

    current = copyNode.next

// Step 2: Set random pointers of copied nodes
current = head

while current != null:
    copyNode = current.next

    if current.random != null:
        copyNode.random = current.random.next

    current = copyNode.next

// Step 3: Restore original list and extract copied list
current = head
newHead = head.next

while current != null:
    copyNode = current.next

    current.next = copyNode.next

    if copyNode.next != null:
        copyNode.next = copyNode.next.next

    current = current.next

return newHead
```

---

## Complexity

```text
Time Complexity:  O(n)
Space Complexity: O(1)
```

The copied nodes are counted as part of the output, not auxiliary space.

---

# Important Edge Cases

## 1. Empty list

```text
head = null
```

Return:

```text
null
```

---

## 2. Single node

```text
1 → null
1.random → 1
```

Return:

```text
1' → null
1'.random → 1'
```

---

## 3. Random pointer is null

```text
1 → 2 → 3
1.random = null
```

Then the copied node should also have:

```text
1'.random = null
```

---

## 4. Random pointer points backward

```text
1 → 2 → 3
2.random → 1
```

This is valid and must be handled correctly.

---

## 5. Random pointer points to itself

```text
1 → 2 → 3
2.random → 2
```

The copied node’s random pointer must point to its own copy, not the original node.

---

# Python Implementation

## Definition

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.random = None
```

---

## O(n) Space Solution Using Hash Map

```python
def copyRandomList(head):
    if not head:
        return None

    old_to_new = {}

    # Create all new nodes
    current = head
    while current:
        old_to_new[current] = Node(current.val)
        current = current.next

    # Assign next and random pointers
    current = head
    while current:
        copy = old_to_new[current]

        if current.next:
            copy.next = old_to_new[current.next]

        if current.random:
            copy.random = old_to_new[current.random]

        current = current.next

    return old_to_new[head]
```

---

## O(1) Extra Space Solution Using Interleaving

```python
def copyRandomList(head):
    if not head:
        return None

    # Step 1: Create copy nodes and insert them after original nodes
    current = head
    while current:
        copy = Node(current.val)
        copy.next = current.next
        current.next = copy
        current = copy.next

    # Step 2: Assign random pointers to copied nodes
    current = head
    while current:
        copy = current.next
        if current.random:
            copy.random = current.random.next
        current = copy.next

    # Step 3: Separate original and copied lists
    newHead = head.next
    current = head

    while current:
        copy = current.next
        current.next = copy.next

        if copy.next:
            copy.next = copy.next.next

        current = current.next

    return newHead
```

---

# Java Implementation

## Definition

```java
class Node {
    int val;
    Node next;
    Node random;

    Node(int val) {
        this.val = val;
    }
}
```

---

## O(1) Extra Space Solution

```java
public class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }

        // Step 1: Create copy nodes after original nodes
        Node current = head;
        while (current != null) {
            Node copy = new Node(current.val);
            copy.next = current.next;
            current.next = copy;
            current = copy.next;
        }

        // Step 2: Assign random pointers to copied nodes
        current = head;
        while (current != null) {
            Node copy = current.next;
            if (current.random != null) {
                copy.random = current.random.next;
            }
            current = copy.next;
        }

        // Step 3: Separate original and copied lists
        Node newHead = head.next;
        current = head;

        while (current != null) {
            Node copy = current.next;

            current.next = copy.next;

            if (copy.next != null) {
                copy.next = copy.next.next;
            }

            current = current.next;
        }

        return newHead;
    }
}
```

---

# C++ Implementation

## Definition

```cpp
class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int value) {
        val = value;
        next = nullptr;
        random = nullptr;
    }
};
```

---

## O(1) Extra Space Solution

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == nullptr) {
            return nullptr;
        }

        // Step 1: Insert copy nodes after original nodes
        Node* current = head;
        while (current != nullptr) {
            Node* copy = new Node(current->val);
            copy->next = current->next;
            current->next = copy;
            current = copy->next;
        }

        // Step 2: Assign random pointers to copied nodes
        current = head;
        while (current != nullptr) {
            Node* copy = current->next;

            if (current->random != nullptr) {
                copy->random = current->random->next;
            }

            current = copy->next;
        }

        // Step 3: Separate original and copied lists
        Node* newHead = head->next;
        current = head;

        while (current != nullptr) {
            Node* copy = current->next;

            current->next = copy->next;

            if (copy->next != nullptr) {
                copy->next = copy->next->next;
            }

            current = current->next;
        }

        return newHead;
    }
};
```

---

# Key Intuition

The hard part is copying random pointers because they do not follow the normal order of the linked list.

The key observation is:

```text
If current.next is the copy of current,
then current.random.next is the copy of current.random.
```

So by placing each copied node immediately after its original node, random pointer relationships become easy to resolve in one pass.

---

# Final Answer

To deep copy a linked list with `next` and `random` pointers:

```text
Best approach:
1. Insert copy nodes after original nodes.
2. Use current.random.next to assign copy.random.
3. Separate the two lists.

Time Complexity:  O(n)
Space Complexity: O(1) auxiliary space
```

This produces a completely new linked list where all `next` and `random` pointers point to copied nodes.

[ Prompt: 107.0 t/s | Generation: 139.5 t/s ]

Exiting...

</details>
</div>

> [Prompt: 107.0 t/s | Generation: 139.5 t/s] - Generated with full GPU offloading on 2x NVIDIA RTX PRO 6000 Blackwell Server Edition using MTP (draft-n-max: 2). GGUF: `Qwen3.8-Flash-Next.MXFP4_MOE.gguf` | Draft MTP: `Qwen3.8-Flash-Next-MTP.BF16.gguf`

## llama.cpp

LLM inference in C/C++ — [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

Quantized using llama.cpp by [prithivMLmods](https://huggingface.co/prithivMLmods).