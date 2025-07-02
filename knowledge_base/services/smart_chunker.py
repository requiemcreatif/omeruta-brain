import os
import re
import logging
import nltk
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class SmartSemanticChunker:
    """Intelligent semantic chunking that preserves context boundaries"""

    def __init__(self):
        # Set Hugging Face token if available
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        self.embedding_model = SentenceTransformer(
            settings.VECTOR_SETTINGS["EMBEDDING_MODEL"],
            use_auth_token=hf_token if hf_token else None,
        )
        self.max_chunk_size = settings.VECTOR_SETTINGS["CHUNK_SIZE"]
        self.overlap_size = settings.VECTOR_SETTINGS["CHUNK_OVERLAP"]
        self.similarity_threshold = 0.75  # For semantic boundary detection

    def chunk_content(self, content: str, page_metadata: Dict = None) -> List[Dict]:
        """Create semantically coherent chunks with quality scoring"""

        if not content or len(content.strip()) < 50:
            return []

        # 1. Clean and preprocess content
        cleaned_content = self._clean_content(content)

        # 2. Split into sentences
        sentences = self._split_into_sentences(cleaned_content)

        if len(sentences) < 2:
            # Single chunk for very short content
            return [self._create_chunk(cleaned_content, 0, page_metadata)]

        # 3. Generate sentence embeddings for boundary detection
        sentence_embeddings = self.embedding_model.encode(sentences)

        # 4. Find semantic boundaries
        boundaries = self._find_semantic_boundaries(sentences, sentence_embeddings)

        # 5. Create chunks respecting boundaries
        chunks = self._create_semantic_chunks(sentences, boundaries, page_metadata)

        # 6. Add quality scores
        for chunk in chunks:
            chunk["quality_score"] = self._calculate_quality_score(chunk["text"])
            chunk["semantic_density"] = self._calculate_semantic_density(chunk["text"])

        return chunks

    def _clean_content(self, content: str) -> str:
        """Clean content while preserving semantic structure"""

        # Remove excessive whitespace but preserve paragraphs
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)

        # Remove navigation elements common in web content
        patterns_to_remove = [
            r"Skip to (?:main )?content",
            r"(?:Home|About|Contact|Privacy|Terms)\s*\|",
            r"Copyright \d{4}",
            r"All rights reserved",
            r"Click here to.*",
            r"Read more.*",
            r"Share this.*",
            r"Tweet\s*Facebook\s*LinkedIn",
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)

        # Clean up remaining artifacts
        content = re.sub(r"\s+", " ", content)
        return content.strip()

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences using NLTK"""

        # Use NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(content)

        # Filter out very short sentences (likely artifacts)
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.isupper():  # Skip ALL CAPS artifacts
                filtered_sentences.append(sentence)

        return filtered_sentences

    def _find_semantic_boundaries(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[int]:
        """Find semantic boundaries between sentences"""

        boundaries = [0]  # Always start with first sentence

        for i in range(1, len(sentences)):
            # Calculate similarity between consecutive sentences
            similarity = cosine_similarity(
                embeddings[i - 1 : i], embeddings[i : i + 1]
            )[0][0]

            # If similarity drops below threshold, it's a boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i)

        boundaries.append(len(sentences))  # Always end with last sentence
        return boundaries

    def _create_semantic_chunks(
        self, sentences: List[str], boundaries: List[int], page_metadata: Dict
    ) -> List[Dict]:
        """Create chunks based on semantic boundaries"""

        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Get sentences for this semantic section
            section_sentences = sentences[start_idx:end_idx]
            section_text = " ".join(section_sentences)

            # Check if section is too large
            section_tokens = len(section_text.split())

            if section_tokens <= self.max_chunk_size:
                # Section fits in one chunk
                chunks.append(
                    self._create_chunk(section_text, len(chunks), page_metadata)
                )
            else:
                # Split large section into smaller chunks
                sub_chunks = self._split_large_section(
                    section_sentences, len(chunks), page_metadata
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_large_section(
        self, sentences: List[str], start_index: int, page_metadata: Dict
    ) -> List[Dict]:
        """Split large semantic sections into smaller chunks"""

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())

            # Check if adding this sentence would exceed limit
            if (
                current_tokens + sentence_tokens > self.max_chunk_size
                and current_chunk_sentences
            ):
                # Create chunk with current sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    self._create_chunk(
                        chunk_text, start_index + len(chunks), page_metadata
                    )
                )

                # Start new chunk with overlap
                overlap_sentences = (
                    current_chunk_sentences[-self.overlap_size :]
                    if self.overlap_size > 0
                    else []
                )
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                self._create_chunk(chunk_text, start_index + len(chunks), page_metadata)
            )

        return chunks

    def _create_chunk(self, text: str, index: int, page_metadata: Dict) -> Dict:
        """Create a chunk dictionary with metadata"""

        return {
            "text": text,
            "index": index,
            "tokens": len(text.split()),
            "metadata": {
                "page_title": page_metadata.get("title", "") if page_metadata else "",
                "page_url": page_metadata.get("url", "") if page_metadata else "",
                "word_count": len(text.split()),
                "char_count": len(text),
                "has_code": "```" in text or "def " in text or "function" in text,
                "has_numbers": bool(re.search(r"\d+", text)),
                "has_urls": bool(re.search(r"https?://", text)),
            },
        }

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate content quality score (0-1)"""

        score = 0.5  # Base score

        # Length scoring
        word_count = len(text.split())
        if 20 <= word_count <= 150:  # Optimal range
            score += 0.2
        elif word_count < 10:  # Too short
            score -= 0.3

        # Sentence structure scoring
        sentences = text.split(".")
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )
        if 8 <= avg_sentence_length <= 25:  # Good sentence length
            score += 0.1

        # Content diversity scoring
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        diversity_ratio = unique_words / total_words if total_words > 0 else 0
        if diversity_ratio > 0.6:  # Good vocabulary diversity
            score += 0.1

        # Penalize low-quality indicators
        if text.count("...") > 3:  # Too many ellipses
            score -= 0.1
        if len(re.findall(r"[A-Z]{2,}", text)) > 5:  # Too many all-caps words
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density - how information-rich the text is"""

        # Simple heuristic based on:
        # - Presence of technical terms
        # - Number density
        # - Proper nouns
        # - Sentence complexity

        density = 0.3  # Base density

        # Technical terms (rough heuristic)
        technical_indicators = [
            "API",
            "algorithm",
            "function",
            "method",
            "system",
            "process",
        ]
        tech_count = sum(
            1 for term in technical_indicators if term.lower() in text.lower()
        )
        density += min(0.3, tech_count * 0.05)

        # Number presence (often indicates concrete information)
        number_count = len(re.findall(r"\d+", text))
        density += min(0.2, number_count * 0.02)

        # Proper nouns (names, places, specific things)
        proper_nouns = len(re.findall(r"\b[A-Z][a-z]+\b", text))
        density += min(0.2, proper_nouns * 0.01)

        return min(1.0, density)
