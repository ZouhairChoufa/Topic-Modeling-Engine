Topic Modeling and TF-IDF Search Engine

This project is a simple search engine that scrapes a predefined list of Wikipedia articles, processes the text, and allows users to search for information using a TF-IDF model. It also uses Latent Dirichlet Allocation (LDA) to identify the main topics within the scraped documents.

Features

Scrapes text content from a list of URLs.

Preprocesses text data: cleaning, lemmatization, and stop-word removal for both French and English.

Builds a TF-IDF (Term Frequency-Inverse Document Frequency) model for information retrieval.

Builds an LDA (Latent Dirichlet Allocation) model for topic modeling.

Provides an interactive command-line interface to:

Search for a query and get a ranked list of relevant documents.

Visualize the TF-IDF similarity scores.

Visualize the topics discovered by the LDA model.

Caches processed data to speed up subsequent runs.

Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites

Python 3.8 or higher

pip for installing packages

Installation

Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>


Create and activate a virtual environment (recommended):

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


Install the required Python packages:

pip install -r requirements.txt


Download the spaCy language models:
This project requires models for French and English to process the text.

python -m spacy download fr_core_news_sm
python -m spacy download en_core_news_sm


Usage

To start the application, run the app.py script from the root directory of the project:

python app.py


The first time you run the script, it will scrape the websites, process the text, and build the models. This may take a few minutes. The processed data will be saved to cached_data.pkl to make future launches much faster.

Once the models are ready, you will be prompted to enter a search query in the console. Type your query and press Enter to see the results.

To exit the program, type exit and press Enter.

Project Structure

.
├── .gitignore
├── LICENSE
├── README.md
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
└── src/
    ├── __init__.py
    ├── data_processing.py # Functions for data fetching and cleaning
    ├── modeling.py        # Functions for TF-IDF and LDA models
    └── visualization.py   # Functions for plotting graphs
