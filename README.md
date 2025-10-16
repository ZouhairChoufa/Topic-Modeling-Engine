Topic Modeling & TF-IDF Search Engine

This project is a command-line search engine built in Python. It scrapes text content from a predefined list of URLs, processes the text, and builds two different models for analysis and search:

TF-IDF (Term Frequency-Inverse Document Frequency): For performing keyword-based document searches and ranking them by relevance.

LDA (Latent Dirichlet Allocation): For identifying the main topics or themes present across the entire collection of documents.

The application allows a user to enter a search query and, in response, it returns the most relevant documents along with a visualization of the underlying topics in the corpus.

Features

Web Scraping: Fetches content from a list of URLs.

Text Processing: Cleans, tokenizes, lemmatizes, and removes stop words from text in both English and French.

TF-IDF Search: Ranks documents based on cosine similarity to a user's query.

LDA Topic Modeling: Discovers abstract topics from the text content.

Data Visualization: Generates plots for TF-IDF scores and LDA topic-word distributions.

Caching: Saves processed data to speed up subsequent runs.

Project Structure

.
├── notebooks/
│   └── Final Code Topic Modling and Tf-IDF.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt


Installation

Clone the repository:

git clone [https://github.com/your-username/Topic-Modeling-Engine.git](https://github.com/your-username/Topic-Modeling-Engine.git)
cd Topic-Modeling-Engine


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required dependencies:

pip install -r requirements.txt


Download SpaCy language models:
The application requires language models for English and French.

python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm


Usage

To run the application, execute the app.py script from the root directory:

python app.py


The script will first attempt to load cached data. If no cache is found, it will scrape the URLs, process the text, and build the models. This initial run may take a few minutes.

Once the models are ready, you will be prompted to enter a search query in the terminal.

Enter your query (or 'exit' to quit): intelligence artificielle


The program will display the search results, followed by plots visualizing the TF-IDF scores and the LDA topics.

Deploying to GitHub

To push this project to a new GitHub repository, follow these steps:

Initialize a local Git repository:

git init


Add all files to staging:

git add .


Commit the files:

git commit -m "Initial commit: Project setup"


Rename the default branch to main:
GitHub's standard is main, while Git's local default might be master.

git branch -M main


Connect to your remote GitHub repository:
Replace the URL with your own repository's URL.

git remote add origin [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)


Push your code to GitHub:

git push -u origin main
