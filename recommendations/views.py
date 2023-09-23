from django.shortcuts import render
from movie.models import Movie

import os
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv, find_dotenv

# Create your views here.
def home(request):
    movies = Movie.objects.all()
    recommended_movies = []
    search_movie = ''

    if request.method == 'POST':
        search_movie = request.POST.get('searchMovie')

        # Load OpenAI API key from environment
        load_dotenv('../openAI.env')
        openai.api_key = os.environ['openAI_api_key']
        emb_search = get_embedding(search_movie, engine='text-embedding-ada-002')
        similarities = [cosine_similarity(list(np.frombuffer(movie.emb)), emb_search) for movie in movies]
        sorted_movies = sorted(zip(movies, similarities), key=lambda x: x[1], reverse=True)
        num_recommendations = 3
        recommended_movies = [movie for movie, _ in sorted_movies[:num_recommendations]]

    return render(request, 'home.html', {
        'movies': recommended_movies,
        'searchTerm': search_movie,
    })