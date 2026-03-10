#!/usr/bin/env python3
"""
Podcast Generator Script

This script implements a hybrid approach to generate a 10-minute podcast summary
from filtered articles. It uses occurrence-based selection, impact scoring,
topic diversity, and narrative flow construction.
"""

import argparse
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys
import os
from collections import defaultdict, Counter
import random
import subprocess
import requests

class PodcastGenerator:
    def __init__(self, articles_file: str = "filtered_articles.txt", config_file: str = None):
        self.articles_file = articles_file
        self.articles = []
        self.selected_articles = []
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Impact keywords and their weights
        self.impact_keywords = self.config.get('impact_scoring', {}).get('keywords', {
            'clinical trial': 5,
            'fda': 5,
            'approval': 5,
            'breakthrough': 4,
            'discovery': 4,
            'first': 4,
            'novel': 4,
            'treatment': 3,
            'cure': 3,
            'drug': 3,
            'therapy': 3,
            'funding': 2,
            'investment': 2,
            'partnership': 2,
            'collaboration': 2,
            'study': 2,
            'research': 2,
            'development': 2
        })
        
        # Topic categories for diversity
        self.topic_categories = self.config.get('topic_categories', {
            'therapeutics': ['treatment', 'therapy', 'drug', 'cure', 'clinical trial', 'fda', 'approval'],
            'diagnostics': ['diagnostic', 'detection', 'screening', 'test', 'biomarker'],
            'research': ['research', 'study', 'discovery', 'breakthrough', 'novel'],
            'industry': ['funding', 'investment', 'partnership', 'collaboration', 'company'],
            'technology': ['technology', 'platform', 'tool', 'device', 'ai', 'machine learning'],
            'genetics': ['gene', 'genetic', 'dna', 'rna', 'genome', 'crispr'],
            'microbiome': ['microbiome', 'bacteria', 'microbial', 'gut', 'microbiome'],
            'cancer': ['cancer', 'oncology', 'tumor', 'carcinoma', 'leukemia'],
            'rare_disease': ['rare disease', 'orphan', 'genetic disorder'],
            'infectious_disease': ['infection', 'virus', 'bacterial', 'pathogen', 'vaccine']
        })
    
    def load_config(self, config_file: str = None) -> Dict:
        """Load configuration from JSON file."""
        default_config_file = "config.json"
        
        # Determine which config file to use
        if config_file is None:
            config_file = default_config_file
        
        # Try to load the config file
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"Loaded configuration from: {config_file}")
                    return config
            except Exception as e:
                print(f"Warning: Could not load config file '{config_file}': {e}")
                print("Using default configuration values.")
        else:
            if config_file != default_config_file:
                print(f"Warning: Config file '{config_file}' not found. Using default configuration values.")
        
        # Return empty dict to use defaults in __init__
        return {}
        
    def parse_articles_file(self) -> List[Dict]:
        """Parse the filtered articles file and extract article information."""
        if not os.path.exists(self.articles_file):
            print(f"Error: Articles file '{self.articles_file}' not found.")
            return []
        
        articles = []
        current_article = {}
        in_metadata = True
        
        try:
            with open(self.articles_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip metadata section
                if line.startswith("ARTICLE DETAILS"):
                    in_metadata = False
                    i += 1
                    continue
                
                if in_metadata:
                    i += 1
                    continue
                
                # Look for article start pattern
                if line.startswith("Article ") and "------------------------------" in lines[i+1]:
                    # Save previous article if exists
                    if current_article:
                        articles.append(current_article)
                    
                    # Start new article
                    current_article = {}
                    article_num = line.split()[1]
                    current_article['number'] = int(article_num)
                    i += 2  # Skip the separator line
                    continue
                
                # Parse article fields
                if line.startswith("Title: "):
                    current_article['title'] = line[7:]
                elif line.startswith("URL: "):
                    current_article['url'] = line[5:]
                elif line.startswith("Source: "):
                    current_article['source'] = line[8:]
                elif line.startswith("Published: "):
                    date_str = line[11:]
                    try:
                        current_article['published_date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        current_article['published_date'] = None
                elif line.startswith("Occurrences: "):
                    current_article['occurrences'] = int(line[13:])
                elif line.startswith("Content Length: "):
                    content_length_str = line[16:]
                    # Remove "characters" suffix if present
                    content_length_str = content_length_str.replace(" characters", "")
                    try:
                        current_article['content_length'] = int(content_length_str)
                    except ValueError:
                        current_article['content_length'] = 0
                elif line.startswith("Content: "):
                    content = line[9:]
                    # Continue reading content if it spans multiple lines
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith("="):
                        content += " " + lines[j].strip()
                        j += 1
                    current_article['content'] = content
                    i = j - 1  # Adjust index
                
                i += 1
            
            # Add the last article
            if current_article:
                articles.append(current_article)
            
            print(f"Successfully parsed {len(articles)} articles from {self.articles_file}")
            return articles
            
        except Exception as e:
            print(f"Error parsing articles file: {e}")
            return []
    
    def calculate_impact_score(self, article: Dict) -> float:
        """Calculate impact score based on keywords and other factors."""
        score = 0.0
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        # Get config values with defaults
        impact_config = self.config.get('impact_scoring', {})
        occurrence_multiplier = impact_config.get('occurrence_multiplier', 2)
        content_length_threshold = impact_config.get('content_length_threshold', 300)
        content_length_bonus = impact_config.get('content_length_bonus', 1)
        recency_bonus = impact_config.get('recency_bonus', {'day1': 2, 'day3': 1})
        
        # Keyword scoring
        for keyword, weight in self.impact_keywords.items():
            if keyword in text:
                score += weight
        
        # Occurrence bonus
        score += article.get('occurrences', 1) * occurrence_multiplier
        
        # Content length bonus (longer articles may have more substance)
        content_length = article.get('content_length', 0)
        if content_length > content_length_threshold:
            score += content_length_bonus
        
        # Recency bonus (more recent articles get slight priority)
        if article.get('published_date'):
            days_old = (datetime.now() - article['published_date']).days
            if days_old <= 1:
                score += recency_bonus.get('day1', 2)
            elif days_old <= 3:
                score += recency_bonus.get('day3', 1)
        
        return score
    
    def classify_topic(self, article: Dict) -> str:
        """Classify article into topic categories."""
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        topic_scores = defaultdict(int)
        
        for topic, keywords in self.topic_categories.items():
            for keyword in keywords:
                if keyword in text:
                    topic_scores[topic] += 1
        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def select_articles_hybrid(self, articles: List[Dict], target_duration: int = None) -> List[Dict]:
        """Select articles using hybrid approach for target duration (in seconds)."""
        if not articles:
            return []
        
        # Get config values with defaults
        selection_config = self.config.get('article_selection', {})
        time_config = self.config.get('time_allocation', {})
        
        # Use target_duration from config if not provided
        if target_duration is None:
            target_duration = self.config.get('target_duration_seconds', 600)
        
        max_main_stories = selection_config.get('max_main_stories', 6)
        max_quick_hits = selection_config.get('max_quick_hits', 12)
        main_stories_candidates = selection_config.get('main_stories_candidates', 10)
        quick_hits_candidates = selection_config.get('quick_hits_candidates', 15)
        max_main_stories_per_topic = selection_config.get('max_main_stories_per_topic', 2)
        max_quick_hits_per_topic = selection_config.get('max_quick_hits_per_topic', 3)
        main_story_duration = selection_config.get('main_story_duration_seconds', 180)
        quick_hit_duration = selection_config.get('quick_hit_duration_seconds', 20)
        enable_temporal_distribution = selection_config.get('enable_temporal_distribution', False)
        temporal_periods = selection_config.get('temporal_periods', 12)  # Default: 12 months
        
        main_stories_percent = time_config.get('main_stories_percent', 0.6)
        quick_hits_percent = time_config.get('quick_hits_percent', 0.3)
        analysis_percent = time_config.get('analysis_percent', 0.1)
        
        # Calculate impact scores and classify topics
        for article in articles:
            article['impact_score'] = self.calculate_impact_score(article)
            article['topic'] = self.classify_topic(article)
        
        # Sort by impact score
        articles.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # If temporal distribution is enabled, organize articles by time periods
        if enable_temporal_distribution:
            # Calculate date range
            dates = [a['published_date'] for a in articles if a.get('published_date')]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                date_range_days = (max_date - min_date).days
                
                if date_range_days > 0:
                    period_duration = date_range_days / temporal_periods
                    
                    # Assign each article to a time period
                    for article in articles:
                        if article.get('published_date'):
                            days_from_start = (article['published_date'] - min_date).days
                            article['temporal_period'] = int(days_from_start / period_duration) if period_duration > 0 else 0
                        else:
                            article['temporal_period'] = -1  # No date
                    
                    print(f"Temporal distribution enabled: {temporal_periods} periods over {date_range_days} days")
        
        selected_articles = []
        topic_coverage = defaultdict(int)
        temporal_coverage = defaultdict(int) if enable_temporal_distribution else None
        estimated_duration = 0
        
        # Target time allocation
        main_stories_time = target_duration * main_stories_percent
        quick_hits_time = target_duration * quick_hits_percent
        analysis_time = target_duration * analysis_percent
        
        # Select main stories
        main_stories = []
        for article in articles[:main_stories_candidates]:
            if len(main_stories) >= max_main_stories:
                break
            
            # Check topic diversity
            topic_ok = topic_coverage[article['topic']] < max_main_stories_per_topic
            
            # Check temporal diversity if enabled
            temporal_ok = True
            if enable_temporal_distribution and temporal_coverage is not None:
                period = article.get('temporal_period', -1)
                if period >= 0:
                    # Allow up to 2 articles per period for main stories
                    max_per_period = max(1, max_main_stories // temporal_periods)
                    temporal_ok = temporal_coverage[period] < max_per_period
                else:
                    # Articles without dates are less preferred
                    temporal_ok = sum(temporal_coverage.values()) < max_main_stories * 0.8
            
            if topic_ok and temporal_ok:
                main_stories.append(article)
                topic_coverage[article['topic']] += 1
                if enable_temporal_distribution and temporal_coverage is not None:
                    period = article.get('temporal_period', -1)
                    if period >= 0:
                        temporal_coverage[period] += 1
                estimated_duration += main_story_duration
        
        # Select quick hits
        quick_hits = []
        remaining_articles = [a for a in articles if a not in main_stories]
        
        for article in remaining_articles[:quick_hits_candidates]:
            if len(quick_hits) >= max_quick_hits:
                break
            
            # Check topic diversity
            topic_ok = topic_coverage[article['topic']] < max_quick_hits_per_topic
            
            # Check temporal diversity if enabled (more lenient for quick hits)
            temporal_ok = True
            if enable_temporal_distribution and temporal_coverage is not None:
                period = article.get('temporal_period', -1)
                if period >= 0:
                    # Allow more articles per period for quick hits
                    max_per_period = max(2, max_quick_hits // temporal_periods)
                    temporal_ok = temporal_coverage[period] < max_per_period
                else:
                    temporal_ok = True  # More lenient for quick hits
            
            if topic_ok and temporal_ok:
                quick_hits.append(article)
                topic_coverage[article['topic']] += 1
                if enable_temporal_distribution and temporal_coverage is not None:
                    period = article.get('temporal_period', -1)
                    if period >= 0:
                        temporal_coverage[period] += 1
                estimated_duration += quick_hit_duration
        
        selected_articles = main_stories + quick_hits
        
        print(f"Selected {len(main_stories)} main stories and {len(quick_hits)} quick hits")
        if enable_temporal_distribution and temporal_coverage:
            periods_covered = len([p for p, count in temporal_coverage.items() if count > 0])
            print(f"Temporal coverage: {periods_covered}/{temporal_periods} periods")
        print(f"Estimated duration: {estimated_duration/60:.1f} minutes")
        
        return selected_articles
    
    def generate_podcast_script(self, articles: List[Dict]) -> str:
        """Generate a podcast script from selected articles."""
        if not articles:
            return "No articles selected for podcast generation."
        
        # Get max_main_stories from config to separate main stories from quick hits
        selection_config = self.config.get('article_selection', {})
        max_main_stories = selection_config.get('max_main_stories', 6)
        
        # Separate main stories and quick hits
        main_stories = articles[:max_main_stories]
        quick_hits = articles[max_main_stories:]
        
        script = []
        
        # Opening
        script.append("=== BIOTECH WEEKLY PODCAST ===")
        script.append("")
        script.append("Welcome to this week's biotech news roundup. I'm your host, and today we're covering the latest developments in biotechnology, from breakthrough discoveries to industry updates.")
        script.append("")
        
        # Main stories section
        script.append("=== MAIN STORIES ===")
        script.append("")
        
        for i, article in enumerate(main_stories, 1):
            script.append(f"Story {i}: {article['title']}")
            script.append("")
            
            # Generate summary
            summary = self.generate_article_summary(article, detailed=True)
            script.append(summary)
            script.append("")
            script.append("---")
            script.append("")
        
        # Quick hits section
        if quick_hits:
            script.append("=== QUICK HITS ===")
            script.append("")
            script.append("Now for some quick updates from around the biotech world:")
            script.append("")
            
            for i, article in enumerate(quick_hits, 1):
                script.append(f"• {article['title']}")
                summary = self.generate_article_summary(article, detailed=False)
                script.append(f"  {summary}")
                script.append("")
        
        # Closing and trends
        script.append("=== TRENDS & INSIGHTS ===")
        script.append("")
        trends = self.analyze_trends(articles)
        script.append(trends)
        script.append("")
        script.append("That wraps up this week's biotech news. Thanks for listening, and we'll see you next week with more updates from the world of biotechnology.")
        script.append("")
        
        # Add source summary
        source_summary = self.generate_source_summary(articles)
        script.append(source_summary)
        
        return "\n".join(script)
    
    def generate_summary_with_ollama(self, article: Dict, detailed: bool = True) -> str:
        """Generate a summary using Ollama with gpt-oss model."""
        title = article.get('title', '')
        content = article.get('content', '')
        
        if not content:
            return "No content available for this article."
        
        # Remove any existing truncation markers
        content = content.replace('...', '').strip()
        
        # Create different prompts based on whether it's detailed or brief
        if detailed:
            # Main stories: 10-15 sentences
            prompt = f"""Please provide a comprehensive, engaging summary of this biotech news article for a podcast. The summary should be 10-15 sentences that thoroughly cover the key points, significance, and implications of the research or development. Write in a conversational tone suitable for a biotech podcast audience. Include details about the research, companies involved, potential impact, and what this means for the industry.

IMPORTANT: Provide ONLY the summary text. Do not include any thinking process, reasoning, or meta-commentary. Just the summary.

Title: {title}

Content: {content}

Summary:"""
        else:
            # Quick hits: 3-5 sentences
            prompt = f"""Please provide a concise, engaging summary of this biotech news article for a podcast. The summary should be 3-5 sentences that capture the key points and significance of the research or development. Write in a conversational tone suitable for a biotech podcast audience.

IMPORTANT: Provide ONLY the summary text. Do not include any thinking process, reasoning, or meta-commentary. Just the summary.

Title: {title}

Content: {content}

Summary:"""
        
        # Log the prompt being sent to Ollama
        print(f"\n=== OLLAMA PROMPT ===")
        print(f"Article: {title}")
        print(f"Prompt: {prompt}")
        print(f"====================\n")
        
        try:
            # Use ollama command line interface
            result = subprocess.run([
                'ollama', 'run', 'gpt-oss:20b', prompt
            ], capture_output=True, text=True, timeout=90)  # Increased timeout for longer summaries
            
            if result.returncode == 0:
                raw_response = result.stdout.strip()
                
                # Log the raw response from Ollama
                print(f"=== OLLAMA RESPONSE ===")
                print(f"Raw response: {raw_response}")
                print(f"======================\n")
                
                # Clean up the response
                summary = raw_response
                
                # Remove common prefixes
                summary = summary.replace('Summary:', '').strip()
                
                # Remove thinking sections (everything between "Thinking..." and "...done thinking.")
                import re
                summary = re.sub(r'Thinking\.\.\..*?\.\.\.done thinking\.', '', summary, flags=re.DOTALL)
                
                # Remove any remaining thinking markers
                summary = re.sub(r'Thinking\.\.\..*', '', summary, flags=re.DOTALL)
                summary = re.sub(r'\.\.\.done thinking\.', '', summary)
                
                # Clean up extra whitespace
                summary = re.sub(r'\n\s*\n', '\n\n', summary).strip()
                
                # Log the cleaned response
                print(f"=== CLEANED SUMMARY ===")
                print(f"Cleaned summary: {summary}")
                print(f"======================\n")
                
                return summary if summary else content[:300] + "..."
            else:
                print(f"Ollama error: {result.stderr}")
                return content[:300] + "..."
                
        except subprocess.TimeoutExpired:
            print("Ollama request timed out")
            return content[:300] + "..."
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return content[:300] + "..."
    
    def generate_article_summary(self, article: Dict, detailed: bool = False) -> str:
        """Generate a summary for an article."""
        title = article.get('title', '')
        content = article.get('content', '')
        
        if detailed:
            # For main stories, use Ollama to generate a comprehensive 10-15 sentence summary
            return self.generate_summary_with_ollama(article, detailed=True)
        else:
            # For quick hits, use Ollama to generate a concise 3-5 sentence summary
            return self.generate_summary_with_ollama(article, detailed=False)
    
    def generate_source_summary(self, articles: List[Dict]) -> str:
        """Generate a summary of all sources used in the podcast."""
        sources = [article.get('source', '') for article in articles]
        source_counts = Counter(sources)
        
        summary = []
        summary.append("=== SOURCES SUMMARY ===")
        summary.append("")
        summary.append("This podcast was compiled from the following sources:")
        summary.append("")
        
        # Sort sources by article count
        sorted_sources = source_counts.most_common()
        
        for source, count in sorted_sources:
            summary.append(f"• {source}: {count} article{'s' if count > 1 else ''}")
        
        summary.append("")
        summary.append(f"Total sources: {len(source_counts)}")
        summary.append(f"Total articles: {len(articles)}")
        
        return "\n".join(summary)
    
    def analyze_trends(self, articles: List[Dict]) -> str:
        """Analyze trends in the selected articles."""
        topics = [article.get('topic', 'general') for article in articles]
        topic_counts = Counter(topics)
        
        # Find most common topics
        top_topics = topic_counts.most_common(3)
        
        # Analyze sources
        sources = [article.get('source', '') for article in articles]
        source_counts = Counter(sources)
        
        trends = "Looking at this week's developments, "
        
        if top_topics:
            main_topic = top_topics[0][0]
            trends += f"the focus has been on {main_topic.replace('_', ' ')} research, "
            
            if len(top_topics) > 1:
                second_topic = top_topics[1][0]
                trends += f"followed by {second_topic.replace('_', ' ')}. "
            else:
                trends += "showing a concentrated effort in this area. "
        
        # Add source diversity comment
        unique_sources = len(source_counts)
        trends += f"We're seeing coverage from {unique_sources} different sources, "
        trends += "indicating broad industry interest in these developments."
        
        return trends
    
    def save_podcast_script(self, script: str, output_file: str):
        """Save the podcast script to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(script)
            
            print(f"Podcast script saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving podcast script: {e}")
    
    def generate_podcast(self, target_duration: int = None) -> str:
        """Generate a complete podcast from articles."""
        # Parse articles
        articles = self.parse_articles_file()
        
        if not articles:
            return "No articles found to generate podcast."
        
        # Select articles using hybrid approach (uses config if target_duration is None)
        selected_articles = self.select_articles_hybrid(articles, target_duration)
        
        if not selected_articles:
            return "No articles selected for podcast generation."
        
        # Generate script
        script = self.generate_podcast_script(selected_articles)
        
        return script

def main():
    """Main function to handle command line arguments and execute podcast generation."""
    parser = argparse.ArgumentParser(
        description="Generate a podcast script from filtered articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python podcast_generator.py
  python podcast_generator.py --duration 480 --output podcast_script.txt
  python podcast_generator.py --input my_articles.txt --duration 720
        """
    )
    
    parser.add_argument('--input', '-i', default='filtered_articles.txt', 
                       help='Input articles file (default: filtered_articles.txt)')
    parser.add_argument('--output', '-o', default='podcast_script.txt',
                       help='Output script file (default: podcast_script.txt)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Target duration in seconds (overrides config, default: use config value)')
    parser.add_argument('--config', '-c', default=None,
                       help='Configuration file path (default: config.json)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PodcastGenerator(args.input, args.config)
    
    # Use duration from args or config
    duration = args.duration
    if duration is None:
        duration = generator.config.get('target_duration_seconds', 600)
    
    print(f"Generating {duration/60:.1f}-minute podcast from {args.input}")
    
    # Generate podcast
    script = generator.generate_podcast(duration)
    
    # Save script
    generator.save_podcast_script(script, args.output)
    
    print("Podcast generation complete!")

if __name__ == "__main__":
    main()
