# app.py — WhatsApp iPhone/Android Export Analyzer (Luxury UI)
import streamlit as st
import pandas as pd
import re
import datetime
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go


try:
    import emoji as _emoji_pkg
    def extract_emojis(text):
        if not text:
            return []
        try:
            return _emoji_pkg.distinct_emoji_list(text)
        except Exception:
            pass
        return re.findall(
            "[" 
            "\U0001F600-\U0001F64F"  
            "\U0001F300-\U0001F5FF"  
            "\U0001F680-\U0001F6FF"  
            "\U0001F1E0-\U0001F1FF"  
            "\U00002700-\U000027BF"  
            "\U00002600-\U000026FF"  
            "]",
            text, flags=re.UNICODE
        )
except Exception:
    def extract_emojis(text):
        if not text:
            return []
        return re.findall(
            "[" 
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U00002600-\U000026FF"
            "]",
            text, flags=re.UNICODE
        )

def extract_links(text):
    if not text:
        return []
    return re.findall(r"http[s]?://\S+", text)

def try_parse_datetime(date_str, time_str):
    candidates = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %I:%M %p",
    ]
    combined = f"{date_str} {time_str}"
    for fmt in candidates:
        try:
            return pd.to_datetime(datetime.datetime.strptime(combined, fmt))
        except Exception:
            continue
    try:
        return pd.to_datetime(combined, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

# ---------- Parser for iOS & Android ----------
def parse_whatsapp_export(text):
    """
    Parses WhatsApp exported chat (iPhone or Android style) into a DataFrame.
    """
    text = text.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    lines = text.splitlines()
    rows = []
    prev = None

    # Patterns for iPhone [date, time] and Android date, time - Sender: message
    patterns = [
        re.compile(r'^\s*\[?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s*,\s*([0-9:\sAPMapm\.]+)\s*\]?\s*(.*)$'),  # iPhone
        re.compile(r'^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),\s*([0-9:\sAPMapm\.]+)\s*-\s*(.*)$')  # Android
    ]

    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if prev is not None:
                prev["message"] += "\n"
            continue
        matched = False
        for pattern in patterns:
            m = pattern.match(line)
            if m:
                date_part = m.group(1).strip()
                time_part = m.group(2).strip()
                rest = m.group(3).strip()
                if ":" in rest:
                    sender, message = re.split(r'\s*:\s*', rest, maxsplit=1)
                    sender = sender.strip()
                    message = message.strip()
                else:
                    sender = "System"
                    message = rest
                dt = try_parse_datetime(date_part, time_part)
                rows.append({"datetime": dt, "sender": sender, "message": message})
                prev = rows[-1]
                matched = True
                break
        if not matched and prev is not None:
            prev["message"] += "\n" + line

    df = pd.DataFrame(rows, columns=["datetime", "sender", "message"])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    df['sender'] = df['sender'].astype(str).str.strip()
    df['message'] = df['message'].astype(str)
    return df

# ---------- Streamlit UI ----------
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0d1117 0%, #0b2030 100%);
    color: #e6eef8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.card {
    background: rgba(15, 25, 45, 0.85);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.7);
}
.small {color: #9aa6b2; font-size: 14px;}
.big-emoji {font-size: 32px;}
h2, h3 {color: #ff6ec7; text-shadow: 0 0 5px #ff6ec7, 0 0 10px #ff6ec7;}
[data-testid="stSidebar"] {background: #0a182f; color: #e6eef8;}
[data-testid="stSidebar"] .stButton>button {background-color: #ff6ec7; color: #fff; border-radius: 8px; border: none;}
[data-testid="stSidebar"] .stButton>button:hover {background-color: #ff4ab8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>WhatsApp Chat Analyzer</h2>
    <div class="small">
       A tool that analyzes WhatsApp chat exports to provide insights on messages, participants, sentiment, emojis, links, and activity patterns.
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:#ff6ec7;text-shadow:0 0 5px #ff6ec7;'>Upload & Filters</h3>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload exported WhatsApp chat (.txt)", type=["txt"])
use_sample = st.sidebar.checkbox("Use sample chat")

sample_text = (
    "[7/1/25, 16:27:06] Alice: Hello from iPhone!\n"
    "7/1/25, 16:28 - Bob: Hello from Android!\n"
    "[01/01/2021, 09:15] Alice: Happy New Year! 🎉\n"
    "[01/01/2021, 09:20:05] Bob: Thanks, same to you! http://example.com/image.png\n"
    "[12/31/20, 10:05:33 PM] Carol: Night folks :)"
)

if use_sample and not uploaded:
    raw = sample_text
elif uploaded:
    raw = uploaded.read().decode('utf-8', errors='ignore')
else:
    st.sidebar.info("Upload an exported WhatsApp chat.txt file or enable the sample chat.")
    st.stop()

df = parse_whatsapp_export(raw)
if df.empty:
    st.error("No messages parsed. Ensure the file is a valid WhatsApp export.")
    st.stop()

df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()
df['word_count'] = df['message'].apply(lambda s: len(re.findall(r"\w+", s)))

participants = sorted(df['sender'].unique().tolist())
participants.insert(0, "All Participants")
selected_participant = st.sidebar.selectbox("Participant", participants)

min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
start_date, end_date = date_range

mask = (df['date'] >= start_date) & (df['date'] <= end_date)
if selected_participant != "All Participants":
    mask &= (df['sender'] == selected_participant)
filtered = df.loc[mask].copy()

main_col, right_col = st.columns([3,1])

# ---------- Summary ----------
with right_col:
    st.markdown('<div class="card"><h3>Summary</h3></div>', unsafe_allow_html=True)
    st.markdown(f"**Messages (shown):** {len(filtered)}")
    st.markdown(f"**Participants (in selection):** {filtered['sender'].nunique()}")
    st.markdown(f"**Words (shown):** {int(filtered['word_count'].sum())}")
    media_count = int(filtered['message'].str.contains('<Media omitted>').sum())
    st.markdown(f"**Media placeholders:** {media_count}")
    st.markdown("---")
    st.markdown('<div class="small">Use participant dropdown and date range to narrow analysis.</div>', unsafe_allow_html=True)

# ---------- Main visualizations ----------
with main_col:
    st.markdown('<div class="card"><h3>Messages per Participant</h3></div>', unsafe_allow_html=True)
    contribution = filtered['sender'].value_counts().reset_index()
    contribution.columns = ['Participant','Messages']
    if not contribution.empty:
        fig = px.bar(contribution, x='Participant', y='Messages', text='Messages', 
                     title='Messages by Participant', template='plotly_dark', color='Messages',
                     color_continuous_scale='RdPu')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No messages for selected filters.")

    st.markdown('<div class="card"><h3>Message Timeline</h3></div>', unsafe_allow_html=True)
    timeline = filtered.groupby('date').size().reset_index(name='count')
    if not timeline.empty:
        fig_t = px.line(timeline, x='date', y='count', title='Messages Over Time', markers=True, template='plotly_dark', line_shape='spline')
        fig_t.update_layout(xaxis_title='Date', yaxis_title='Messages')
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Not enough data to render timeline.")

    st.markdown('<div class="card"><h3>Word Cloud</h3></div>', unsafe_allow_html=True)
    combined = " ".join(filtered['message'].astype(str).tolist())
    combined = re.sub(r'http\S+', '', combined)
    combined = re.sub(r'<Media omitted>', '', combined)
    if combined.strip():
        wc = WordCloud(width=900, height=350, background_color='black', colormap='plasma').generate(combined)
        fig_wc = plt.figure(figsize=(12,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("No textual content for word cloud.")

    st.markdown('<div class="card"><h3>Emoji Usage</h3></div>', unsafe_allow_html=True)
    all_emojis = []
    for m in filtered['message'].astype(str):
        all_emojis.extend(extract_emojis(m))
    if all_emojis:
        em_counts = Counter(all_emojis).most_common(15)
        em_df = pd.DataFrame(em_counts, columns=['Emoji','Count'])
        em_df_display = em_df.copy()
        em_df_display['Emoji'] = em_df_display['Emoji'].apply(lambda e: f"<span class='big-emoji'>{e}</span>")
        st.write(em_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        fig_em = px.bar(em_df, x='Emoji', y='Count', title='Emoji Frequency', template='plotly_dark', color='Count', color_continuous_scale='PuRd')
        st.plotly_chart(fig_em, use_container_width=True)
    else:
        st.info("No emojis found in selection.")

    st.markdown('<div class="card"><h3>Links Shared </h3></div>', unsafe_allow_html=True)
    links = []
    for _, r in filtered.iterrows():
        for l in extract_links(r['message']):
            links.append({'Timestamp': r['datetime'].strftime("%Y-%m-%d %H:%M:%S"), 'Sender': r['sender'], 'URL': l})
    if links:
        links_df = pd.DataFrame(links)
        links_df['Open'] = links_df['URL'].apply(lambda u: f"<a href='{u}' target='_blank'>{u}</a>")
        st.write(links_df[['Timestamp','Sender','Open']].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No links detected in selection.")

    st.markdown('<div class="card"><h3>Sentiment Analysis</h3></div>', unsafe_allow_html=True)
    analyzer = SentimentIntensityAnalyzer()
    if not filtered.empty:
        filtered['compound'] = filtered['message'].apply(lambda t: analyzer.polarity_scores(str(t))['compound'])
        filtered['Sentiment'] = filtered['compound'].apply(lambda c: 'Positive' if c>0.05 else ('Negative' if c<-0.05 else 'Neutral'))
        sent_summary = filtered['Sentiment'].value_counts().reset_index()
        sent_summary.columns = ['Sentiment','Count']
        fig_s = px.pie(sent_summary, values='Count', names='Sentiment', title='Sentiment Distribution', template='plotly_dark', color_discrete_sequence=['#ff4ab8','#9aa6b2','#4adfff'])
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown("""**Explanation:** In this Positive messages have an overall positive tone;
                     Neutral messages show no strong polarity; 
                    Negative messages lean negative. """, unsafe_allow_html=True)
    else:
        st.info("No messages available for sentiment analysis.")

    st.markdown('<div class="card"><h3>Activity Heatmap (Hour × Day)</h3></div>', unsafe_allow_html=True)
    if not filtered.empty:
        heat = pd.crosstab(filtered['hour'], filtered['day']).reindex(index=list(range(0,24)), fill_value=0)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        present_days = [d for d in day_order if d in heat.columns]
        heat = heat.reindex(columns=present_days, fill_value=0)
        if present_days:
            fig_h = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Viridis'))
            fig_h.update_layout(title='Messages by Hour (y) and Day (x)', xaxis_title='Day', yaxis_title='Hour', template='plotly_dark')
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Insufficient day-of-week information for heatmap.")
    else:
        st.info("No data for heatmap.")

    st.markdown('<div class="card"><h3>Top Words</h3></div>', unsafe_allow_html=True)
    topw = (lambda t: Counter(re.findall(r"\w+", t.lower())).most_common(25))(combined)
    if topw:
        wdf = pd.DataFrame(topw, columns=['Word','Count'])
        fig_w = px.bar(wdf, x='Word', y='Count', title='Top Words', template='plotly_dark', color='Count', color_continuous_scale='PuBu')
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("No prominent words detected.")

    st.markdown('<div class="card"><h3>Recent Messages </h3></div>', unsafe_allow_html=True)
    recent = filtered[['datetime','sender','message','word_count']].sort_values('datetime', ascending=False).head(200)
    recent_display = recent.copy()
    recent_display['datetime'] = recent_display['datetime'].astype(str)
    st.dataframe(recent_display.reset_index(drop=True))

st.markdown("---")
st.markdown('<div class="small"> My 3rd sem Mini Project using python & ML concepts @Sejalpotdar05@gmail.com</div>', unsafe_allow_html=True)
