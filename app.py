import re
import sqlite3
import warnings
from functools import lru_cache

import joblib
import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

warnings.filterwarnings('ignore')



app = Flask(__name__)
model = joblib.load('model.sav')
model1 = joblib.load('model1.sav')
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

POSITIVE_WORDS = {
    'hope', 'better', 'healing', 'recover', 'recovery', 'support', 'safe', 'calm',
    'peace', 'joy', 'grateful', 'love', 'strong', 'improving', 'relief', 'help',
    'connected', 'smile', 'optimistic', 'okay', 'good'
}
NEGATIVE_WORDS = {
    'suicide', 'selfharm', 'self-harm', 'kill', 'die', 'death', 'hopeless', 'empty',
    'worthless', 'alone', 'pain', 'hurt', 'depressed', 'depression', 'anxious',
    'anxiety', 'broken', 'tired', 'unhappy', 'cry', 'cut', 'bleed', 'end'
}
AMBIGUOUS_WORDS = {
    'maybe', 'perhaps', 'unsure', 'confused', 'numb', 'idk', 'unclear', 'mixed',
    'whatever', 'fine', 'nothing', 'different'
}
ANGER_WORDS = {
    'angry', 'mad', 'furious', 'annoyed', 'rage', 'hate', 'irritated', 'frustrated'
}
DISGUST_WORDS = {
    'disgust', 'gross', 'nasty', 'revolting', 'sickened', 'awful'
}
FEAR_WORDS = {
    'fear', 'scared', 'terrified', 'panic', 'worried', 'afraid', 'nervous'
}
JOY_WORDS = {
    'happy', 'joy', 'excited', 'delighted', 'grateful', 'glad', 'smile', 'relieved'
}
SADNESS_WORDS = {
    'sad', 'down', 'crying', 'cry', 'lonely', 'miserable', 'grief', 'hopeless'
}
SURPRISE_WORDS = {
    'surprised', 'shocked', 'suddenly', 'unexpected', 'wow', 'astonished'
}
SELF_HARM_TOPIC_WORDS = {
    'suicide', 'selfharm', 'self-harm', 'cut', 'bleed', 'die', 'death', 'kill',
    'worthless', 'hopeless', 'overdose', 'endit', 'end', 'injure', 'hurtmyself'
}
HIGH_RISK_PHRASES = {
    'kill myself', 'want to die', 'end my life', 'hurt myself', 'self harm',
    'self-harm', 'no reason to live', 'better off dead', 'want to disappear',
    'suicidal thoughts', 'thinking about death'
}

SENTIMENT_ANCHORS = {
    'positive': [
        'I feel hopeful, supported, and optimistic about life.',
        'I am healing, safe, and thankful for the people around me.',
    ],
    'negative': [
        'I feel hopeless, alone, broken, and emotionally overwhelmed.',
        'Everything feels painful, empty, and hard to continue.',
    ],
    'ambiguous': [
        'I feel mixed, confused, unsure, and emotionally unclear.',
        'My thoughts are uncertain and difficult to describe.',
    ],
    'neutral': [
        'This is an ordinary day and I am describing events without much emotion.',
        'I am sharing information in a calm and factual way.',
    ],
}

EMOTION_ANCHORS = {
    'anger': [
        'I feel angry, frustrated, irritated, and full of rage.',
    ],
    'disgust': [
        'I feel disgusted, sickened, and revolted by everything around me.',
    ],
    'fear': [
        'I feel scared, afraid, anxious, nervous, and unsafe.',
    ],
    'joy': [
        'I feel joyful, relieved, happy, and grateful today.',
    ],
    'sadness': [
        'I feel deeply sad, lonely, empty, and I want to cry.',
        'There is no comfort, connection, or hope in my life.',
    ],
    'surprise': [
        'I feel shocked, surprised, and overwhelmed by sudden events.',
    ],
    'neutral': [
        'My emotions are steady, muted, and mostly neutral.',
    ],
}

TOPIC_ANCHORS = {
    'non_self_harm': [
        'This text is about daily life, work, study, family, and ordinary stress.',
        'The topic does not involve self-harm, suicide, or injury.',
    ],
    'self_harm': [
        'This text is about suicide, self-harm, death, hurting myself, and wanting to disappear.',
        'The writer questions the purpose of life and expresses self-destructive thoughts.',
    ],
}

RISK_ANCHORS = {
    'low': [
        'I feel okay, safe, and connected even if I have some stress.',
    ],
    'moderate': [
        'I feel sad, alone, and unsupported, and I do not know who to talk to.',
    ],
    'high': [
        'I feel like life has no purpose and I do not know why I am living.',
        'I feel trapped, hopeless, and emotionally exhausted with no support.',
    ],
    'critical': [
        'I want to die, hurt myself, and I am thinking about ending my life.',
        'I am actively thinking about suicide and self-harm right now.',
    ],
}


def _count_matches(tokens, vocabulary):
    return sum(1 for token in tokens if token in vocabulary)


def _contains_phrase(text, phrases):
    lowered = text.lower()
    return sum(1 for phrase in phrases if phrase in lowered)


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=None)
def _encode_text_cached(text):
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=True)


def _semantic_profile(text, anchors):
    text_embedding = _encode_text_cached(text)
    scores = []
    for label, examples in anchors.items():
        example_embeddings = [_encode_text_cached(example) for example in examples]
        similarities = [float(cos_sim(text_embedding, example_embedding)[0][0]) for example_embedding in example_embeddings]
        scores.append(max(similarities))

    scores_array = np.array(scores, dtype=float)
    shifted = np.exp((scores_array - scores_array.max()) * 4.0)
    normalized = shifted / shifted.sum()
    return normalized.tolist()


def extract_text_features(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]

    semantic_sentiment = np.array(_semantic_profile(text, SENTIMENT_ANCHORS))
    semantic_emotion = np.array(_semantic_profile(text, EMOTION_ANCHORS))
    semantic_topic = np.array(_semantic_profile(text, TOPIC_ANCHORS))

    positive = _count_matches(tokens, POSITIVE_WORDS)
    negative = _count_matches(tokens, NEGATIVE_WORDS)
    ambiguous = _count_matches(tokens, AMBIGUOUS_WORDS)
    anger = _count_matches(tokens, ANGER_WORDS)
    disgust = _count_matches(tokens, DISGUST_WORDS)
    fear = _count_matches(tokens, FEAR_WORDS)
    joy = _count_matches(tokens, JOY_WORDS)
    sadness = _count_matches(tokens, SADNESS_WORDS)
    surprise = _count_matches(tokens, SURPRISE_WORDS)
    self_harm_topic = _count_matches(tokens, SELF_HARM_TOPIC_WORDS)
    phrase_hits = _contains_phrase(text, HIGH_RISK_PHRASES)

    lexical_sentiment = np.array([
        0.2 + positive,
        0.2 + negative + (self_harm_topic * 0.6) + (phrase_hits * 0.8),
        0.2 + ambiguous,
        0.2,
    ], dtype=float)
    lexical_sentiment = lexical_sentiment / lexical_sentiment.sum()

    lexical_emotion = np.array([
        0.2 + anger,
        0.2 + disgust,
        0.2 + fear + (phrase_hits * 0.3),
        0.2 + joy,
        0.2 + sadness + (self_harm_topic * 0.4) + (phrase_hits * 0.5),
        0.2 + surprise,
        0.2,
    ], dtype=float)
    lexical_emotion = lexical_emotion / lexical_emotion.sum()

    lexical_topic = np.array([
        0.2,
        0.2 + self_harm_topic + (phrase_hits * 1.5),
    ], dtype=float)
    lexical_topic = lexical_topic / lexical_topic.sum()

    sentiment_vector = (semantic_sentiment * 0.7) + (lexical_sentiment * 0.3)
    emotion_vector = (semantic_emotion * 0.7) + (lexical_emotion * 0.3)
    topic_vector = (semantic_topic * 0.65) + (lexical_topic * 0.35)

    features = np.concatenate([sentiment_vector, emotion_vector, topic_vector])
    return features.tolist()


def assess_crisis_risk(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    semantic_risk = _semantic_profile(text, RISK_ANCHORS)
    negative_hits = _count_matches(tokens, NEGATIVE_WORDS)
    fear_hits = _count_matches(tokens, FEAR_WORDS)
    sadness_hits = _count_matches(tokens, SADNESS_WORDS)
    self_harm_hits = _count_matches(tokens, SELF_HARM_TOPIC_WORDS)
    phrase_hits = _contains_phrase(text, HIGH_RISK_PHRASES)

    # Weight explicit self-harm language heavily so obviously risky text
    # cannot still look nominal after feature normalization.
    risk_score = (
        (negative_hits * 1.2)
        + (fear_hits * 0.8)
        + (sadness_hits * 0.8)
        + (self_harm_hits * 3.0)
        + (phrase_hits * 4.0)
    )

    if semantic_risk[3] >= 0.35 or phrase_hits >= 1 or self_harm_hits >= 3 or risk_score >= 12:
        return 'critical'
    if semantic_risk[2] >= 0.33 or self_harm_hits >= 2 or risk_score >= 8:
        return 'high'
    if semantic_risk[1] >= 0.32 or negative_hits >= 2 or risk_score >= 4:
        return 'moderate'
    return 'low'


def calculate_semantic_adjustment(text, extracted_features):
    risk_profile = _semantic_profile(text, RISK_ANCHORS)
    self_harm_topic = extracted_features[12]
    negative_sentiment = extracted_features[1]
    sadness = extracted_features[8]
    fear = extracted_features[6]

    low_risk = risk_profile[0]
    moderate_risk = risk_profile[1]
    high_risk = risk_profile[2]
    critical_risk = risk_profile[3]

    severity = (
        (moderate_risk * 0.2)
        + (high_risk * 0.8)
        + (critical_risk * 1.4)
        + (self_harm_topic * 0.9)
        + (negative_sentiment * 0.25)
        + (sadness * 0.2)
        + (fear * 0.15)
        - (low_risk * 0.85)
    )
    severity = max(0.0, severity)

    death_adjustment = min(140.0, severity * 42.0)
    injury_adjustment = min(210.0, severity * 64.0)
    return death_adjustment, injury_adjustment


@app.route('/predict',methods=['POST'])
def predict():
    text_input = request.form.get('text_input', '').strip()
    if not text_input:
        return render_template('home.html', message='Please enter some text to analyze.')

    extracted_features = extract_text_features(text_input)
    final_features = [np.array(extracted_features)]
    predict = model.predict(final_features)
    predict1 = model1.predict(final_features)
    crisis_risk = assess_crisis_risk(text_input)
    death_adjustment, injury_adjustment = calculate_semantic_adjustment(text_input, extracted_features)

    death_output = float(predict[0]) + death_adjustment
    injury_output = float(predict1[0]) + injury_adjustment

    # Floor the outputs when explicit crisis language is present.
    if crisis_risk == 'critical':
        death_output = max(death_output, 380.0)
        injury_output = max(injury_output, 930.0)
    elif crisis_risk == 'high':
        death_output = max(death_output, 280.0)
        injury_output = max(injury_output, 700.0)
    elif crisis_risk == 'moderate':
        death_output = max(death_output, 180.0)
        injury_output = max(injury_output, 500.0)
    else:
        death_output = min(death_output, 150.0)
        injury_output = min(injury_output, 430.0)

    feature_labels = [
        ('MS-Positive', extracted_features[0]),
        ('MS-Negative', extracted_features[1]),
        ('MS-Ambiguous', extracted_features[2]),
        ('MS-Neutral', extracted_features[3]),
        ('ME-Anger', extracted_features[4]),
        ('ME-Disgust', extracted_features[5]),
        ('ME-Fear', extracted_features[6]),
        ('ME-Joy', extracted_features[7]),
        ('ME-Sadness', extracted_features[8]),
        ('ME-Surprise', extracted_features[9]),
        ('ME-Neutral', extracted_features[10]),
        ('M-NonST Topic', extracted_features[11]),
        ('M-ST Topic', extracted_features[12]),
    ]

    return render_template(
        'prediction.html',
        output=death_output,
        output1=injury_output,
        input_text=text_input,
        feature_labels=feature_labels,
        crisis_risk=crisis_risk
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


if __name__ == "__main__":
    app.run(debug=True)
