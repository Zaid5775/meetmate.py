from flask import Flask, request, jsonify
import nltk
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.tokenize import sent_tokenize, word_tokenize

import random
# nltk.download('stopwords')
# from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
from docx import Document
import requests

# audio video
import io
from werkzeug.utils import secure_filename
import tempfile
import os
import pysubs2
import cv2
import time
import webvtt
import mutagen
import whisper
import datetime
import moviepy.editor 
from mutagen.mp3 import MP3
from mutagen.ogg import OggFileType
from mutagen.wave import WAVE
from pydub import AudioSegment
# import speech_recognition as sr
from pydub.utils import make_chunks








app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

CHUNK_SIZE = 1000
MODEL_NAME =  "knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI"

def key_points(summary):
    # Tokenize the summary into words
    words = summary.split()

# Join the words back into a string
    text = " ".join(words)

# Create a TF-IDF vectorizer
    # stop_words = ['s', 'your', 'we', 'then', 'our', "couldn't", 'shouldn', 'these', "that'll", 'can', 'each', 'theirs', 'between', "didn't", 'above', 'doesn', 'any', 'to', 'about', 'during', 'be', 'him', "haven't", 'himself', "you'd", 'down', 'do', 'where', 'than', 'themselves', 'their', 'yours', 'd', 'me', 've', 'before', 'an', 'at', 'won', 'just', 'he', 'does', "should've", 'i', 'with', 'she', 'other', 'wasn', 'which', 'against', 'has', 'yourself', 'as', "won't", "doesn't", 'further', 'both', 'o', 'ours', 'it', 'too', 'his', 'out', 'until', 'you', 'was', 'more', 'isn', 'were', "needn't", 'the', 'being', 'if', 'when', 'there', 'most', 'because', 'don', 'ma', 'haven', 'same', 'y', 'here', 'that', 'hers', 'off', 'few', "hadn't", 'a', 'all', 'hadn', 'those', 'only', 'will', 'or', "you're", 'are', 'now', "mustn't", "shan't", 'shan', 'such', 'should', 'not', 'why', 'once', 'my', 'for', 'ain', 'on', 'been', 'but', 'hasn', 'herself', 'aren', "wasn't", 'mustn', 'is', "she's", 'under', 'so', 'did', 'while', "shouldn't", "you've", "hasn't", 'they', 'have', 'her', 'some', 'nor', 'by', 'after', 'this', 'what', 't', 'whom', 'again', 'had', 'from', 'very', "aren't", 'no', 'and', 'over', 'didn', 'in', 'wouldn', 'of', 'yourselves', "weren't", 'mightn', 'its', "wouldn't", "don't", "it's", 'm', 're', 'll', 'below', 'through', 'ourselves', 'weren', 'up', "you'll", 'needn', 'myself', 'how', 'doing', 'into', "isn't", 'couldn', 'who', 'am', "mightn't", 'own', 'itself', 'having', 'them']
    stop_words = ["0o", "0s",'started', "forward", "sample" , "check", "meeting" , "talk", "talking","work", "glad",  "discuss","person","tells" , 'good', 'works' ,'reaching','estimated','entire','measure', '80' , '20' , '86', 'andthe', "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
    
    vectorizer = TfidfVectorizer(stop_words=stop_words)

# Compute the TF-IDF scores
    scores = vectorizer.fit_transform([text])

# Get the feature names (i.e., the words)
    # feature_names = vectorizer.get_feature_names()
    feature_names = vectorizer.vocabulary_.keys()

# Get the TF-IDF scores for each word
    tfidf_scores = scores.toarray()[0]

# Create a list of (word, score) pairs
    word_scores = [(word, score) for word, score in zip(feature_names, tfidf_scores)]

# Sort the (word, score) pairs by score in descending order
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

# Get the top 10 words
    top_words = [word for word, score in word_scores[:20]]
    
    return top_words

def chunk_text(text):
    chunks = []
    while len(text) > CHUNK_SIZE:
        chunk = text[:CHUNK_SIZE]
        text = text[CHUNK_SIZE:]
        chunks.append(chunk)
    chunks.append(text)
    return chunks

def extract_text_from_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

API_URL = "https://api-inference.huggingface.co/models/knkarthick/TOPIC-SUMMARY"
headers = {"Authorization": "Bearer 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"}


# total no
def att(summary):
    nlp = spacy.load("en_core_web_sm")
    


    count = 0
    attendees = set()
    
    for ent in nlp(summary).ents:
        if ent.label_ == "PERSON":
            attendees.add(ent.text)
    count = len(attendees)
    print(attendees)
    print(count)
    return count
    
def query(payload):

        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()[0]['generated_text']
        return output



tokenizer = AutoTokenizer.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)










# #FASTER
# def txt_from_audio_whisper_model (file_path):
#     # try:
#     #     folder = "../Materials/audio_{}".format (file_path.split ("/") [-1].split (".") [0])
#     #     os.makedirs (folder)
#     #     os.open ("{}/{}.{}".format (folder, file_path.split ("/") [-1].split ('.') [0]), file_path.split ("/") [-1].split ('.') [-1], "w+")
#     # except:
#     #     pass
    
#     # txt_file = open ("../Materials/Transcript/transcript_{}.txt".format (file_path.split ("/") [-1].split (".") [0]), "w+")
#     model = whisper.load_model ('base')
#     result = model.transcribe (file_path, fp16=False)
#     # txt_file.write (result['text'] + "\n")
    
#     return result['text']







def extract_text_from_vtt(vtt_file_path):
    
    vtt = webvtt.read(vtt_file_path)
    transcript = ""

    lines = []
    for line in vtt:
        lines.extend(line.text.strip().splitlines())

    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line

    return transcript.strip()


def txt_from_audio_whisper_model(file_path):
    model = whisper.load_model('base')
    result = model.transcribe(file_path, fp16=False)
    print(result['text'])
    return result['text']

def mp4_to_mp3(video_path):
    mp3 = "./Materials/Audio/{}.mp3".format(video_path.split("/")[-1].split(".")[0])
    video = moviepy.editor.VideoFileClip(video_path)
    audio_from_video = video.audio
    audio_from_video.write_audiofile(mp3)
    print('MP4 TO MP3 DONE')
    return mp3

def mp3_to_wav(path_mp3):
    sound = AudioSegment.from_file(path_mp3)
    sound.export("./Materials/Audio/{}.wav".format(path_mp3.split("/")[-1].split(".")[0]), format="mp3", bitrate="128k")
    path_wav = "./Materials/Audio/{}.wav".format(path_mp3.split("/")[-1].split(".")[0])
    audio_mp3 = AudioSegment.from_mp3("./Materials/Audio/{}.wav".format(path_mp3.split("/")[-1].split(".")[0]))
    audio_mp3.export(path_wav, format="wav")
    print("MP3 TO WAV DONE")
    return path_wav

def ogg_to_wav(file):
    path_ogg = "./Materials/Audio/{}".format(file.filename)
    file.save(path_ogg) # save the file to disk
    path_wav = "./Materials/Audio/{}.wav".format(path_ogg.split("/")[-1].split(".")[0])
    audio_ogg = AudioSegment.from_ogg(path_ogg)
    audio_ogg.export(path_wav, format="wav")
    print('OGG TO WAV, DONE .....')
    



def mp4_len (file):
    data = cv2.VideoCapture (file)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    mp4_len_dic = {
        "video_frames": frames,
        "video_speed_fps": fps,
        "video_len_in_secs": seconds,
        "video_time": video_time
    }
    return mp4_len_dic

#Audio length
def audio_len (file):
    audio = WAVE (file)
    audio_info = audio.info
    length = int (audio_info.length)
    hrs = length // 3600
    length %= 3600
    mins = length // 60
    length %= 60
    secs = length
    audio_time = {
        "hours": hrs,
        "minutes": mins,
        "secs": secs
    }
    return audio_time

@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    if request.method != 'POST':
        return "Method not allowed", 405
    else:
        file = request.files['file']
        filename = file.filename
        
        # temp_dir = tempfile.gettempdir()
        # path = os.path.join(temp_dir, vtt_file.filename)
        # file.save(path)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        if filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file)
        elif filename.endswith('.vtt'):
             file.save(temp_file.name)
             text = extract_text_from_vtt(temp_file.name)
           

        elif filename.endswith('.mp4'):
            mp3_path = mp4_to_mp3(file)
            wav_path = mp3_to_wav(mp3_path)
            text = txt_from_audio_whisper_model(wav_path)
        elif filename.endswith('.mp3'):
            wav_path = mp3_to_wav(file)
            text = txt_from_audio_whisper_model(wav_path)
        elif filename.endswith('.mpeg'):
            wav_path = mp3_to_wav(file)
            text = txt_from_audio_whisper_model(wav_path)
        elif filename.endswith('.ogg'):
            wav_path = ogg_to_wav(file)
            text = txt_from_audio_whisper_model(wav_path)
        elif filename.endswith('.wav'):
            text = txt_from_audio_whisper_model(file)
        else:
            return "Unsupported file format", 400
        
        temp_file.close()
        os.unlink(temp_file.name)

       
        
        summary_length = request.form['summary_length']
        print(summary_length)
        
        # snt = 0
        # if summary_length == 'short':
        #     snt = 50
        #     minn = 30
        #     maxx = 30
        # elif summary_length == 'medium':
        #     snt = 50
        #     minn = 40
        #     maxx = 40
        
        # elif summary_length == 'long':
        #     minn = 60
        #     maxx = 60
        #     snt =55
        # else:
        #     snt = 1000
        if summary_length == 'short':
            maxx = 50
            max_sent_len = 15
            num_sentences = 7
        elif summary_length == 'medium':
            maxx = 100
            max_sent_len = 17
            num_sentences = 10
        else:
            maxx = 120
            max_sent_len = 13
            num_sentences = 15
           
           
           
        # parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Create a summarizer using the LexRank algorithm
        # summarizer = LexRankSummarizer()
    
    # Summarize the text
        # summary_text = summarizer(parser.document, snt) # 5 is number of sentences to extract
        # print(summary_text)
    # Convert the summary to a string
        # new_sentences = ' '.join([str(sentence) for sentence in text]) 
        # inputs = tokenizer(text, return_tensors="pt").input_ids
        # model = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
        # outputs = model.generate(inputs, max_length=maxx, min_length=minn, do_sample=False)
        # summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        
        # main  ...
        # chunks = chunk_text(text)
        # results = []
        # for chunk in chunks:
        #     inputs = tokenizer(chunk, return_tensors="pt").input_ids
            
        #     outputs = model.generate(inputs, do_sample=False,max_length=maxx, min_length=minn)
        #     results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # summary = ''.join(results)
        
        max_length = 1000
        # Generate summaries for each chunk of text
        chunks = chunk_text(text)
        results = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt").input_ids
            outputs = model.generate(inputs, do_sample=False, max_length=maxx)
            decoded_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary_sentences = nltk.sent_tokenize(decoded_summary)
            summary = ''
            for sentence in summary_sentences:
                if len(summary) + len(sentence) <= num_sentences * max_sent_len:
                    summary  += sentence.strip() + ' '
                else:
                    break
            results.append(summary)
        summary = ''.join(results)
        
	#     "inputs": text,
    # })
        # print("hi")
        # print(output)
        # print("hi")
        # summary = output
       
       
       
       
       
       
        # generated_text = query({
	    # "inputs": summary,
        # })
     
        
        
        bullet = '\n- '.join([str(sentence) for sentence in summary])
        bullet_points =  f"- {bullet}"
        # summary = ' '.join([str(sentence) for sentence in summary_text])
        
        original_sentences = sent_tokenize(text)
        original_word_count = 0
        for sentence in original_sentences:
            original_word_count += len(word_tokenize(sentence))
        original_char_count = len(text)
        
        summarized_sentences = sent_tokenize(summary)
        summarized_word_count = 0
        for sentence in summarized_sentences:
            summarized_word_count += len(word_tokenize(sentence))
        summarized_char_count = len(summary)
        
     
        
        
        # print(generated_text)
        key = key_points(summary)
        
        atts = att(summary)
       
        return jsonify({'summary': summary,
            'bullet_points' : bullet_points,
            'original_sentence_count': len(original_sentences),
            'original_word_count': original_word_count,
            'original_char_count': original_char_count,
            'summarized_sentence_count': len(summarized_sentences),
            'summarized_word_count': summarized_word_count,
            'key' : key,
            'att' : atts,
            'summarized_char_count': summarized_char_count,
            # 'topic' : generated_text,
                
        })  

if __name__ == "__main__":
    app.run(debug= True)











 



# import memory_profiler

# # Use smaller models
# model = create_small_model()

# # Use only necessary components
# tokenizer = load_tokenizer()

# def process_data(data):
#     # Use memory-efficient data structures
#     processed_data = (preprocess_data(datum) for datum in data)

#     # Reduce chunk size
#     chunk_size = 1000
#     for i in range(0, len(processed_data), chunk_size):
#         chunk = processed_data[i:i + chunk_size]

#         # Use caching
#         cached_results = cache_results(chunk)

#         # Use parallel processing
#         results = process_in_parallel(cached_results)

# # Monitor memory usage
# memory_profiler.run("process_data(data)")

# # Optimize the code
# def process_in_parallel(cached_results):
#     optimized_results = []
#     for result in cached_results:
#         optimized_result = optimize_result(result)
#         optimized_results.append(optimized_result)
#     return optimized_results
