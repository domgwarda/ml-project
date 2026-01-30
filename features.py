import re
import tldextract
import pandas as pd

FEATURES = []

def feature(fn):
    FEATURES.append(fn)
    return fn

@feature
def length(URL: pd.Series) -> pd.Series:
    return URL.str.len()

@feature
def slash_num(URL):
    return URL.str.count("/")

@feature
def dot_num(URL):
    return URL.str.count(".")

@feature
def dash_num(URL):
    return URL.str.count("-")

@feature
def https(URL):
    return URL.str.startswith("https").astype(int)

@feature
def http(URL):
    return URL.str.startswith("http:").astype(int)

@feature
def digits_in_url_num(URL):
    return URL.str.findall(r'\d').str.len()
    
_extractor = tldextract.TLDExtract(cache_dir=True)

def _ext_series(URL):
    return URL.apply(_extractor)

def _split(val: str):
    return [x for x in re.split(r'[.-]', val) if x]

def subdomain(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: set(_split(x.subdomain)))

def domains(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: set(_split(x.domain)))

def suffix(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: set(_split(x.suffix)))
    
@feature
def subdomain_num(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: len(_split(x.subdomain)))

@feature
def domains_num(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: len(_split(x.domain)))

@feature
def suffix_num(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: len(_split(x.suffix)))

@feature
def digits_in_domain(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: sum(c.isdigit() for c in x.domain))

@feature
def digits_in_subdomain(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: sum(c.isdigit() for c in x.subdomain))

@feature
def digits_in_suffix(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: sum(c.isdigit() for c in x.suffix))

@feature
def is_www(URL):
    return (subdomain(URL) == {'www'}).astype(int)

@feature
def is_com(URL):
    return (suffix(URL) == {'com'}).astype(int)

@feature
def has_com(URL):
    return suffix(URL).apply(lambda x: int('com' in x))

popular_websites_df = pd.read_csv('OtherData/Web_Scrapped_websites.csv', encoding='latin1')[['Website']]

ext_popular = popular_websites_df['Website'].apply(_extractor)

popular_domains = (ext_popular.apply(lambda x: _split(x.domain)).explode().dropna().unique())

POPULAR_DOMAINS = set(popular_domains)
POPULAR_DOMAINS.discard('www')

@feature
def popular_domain_in_domains(URL):
    dom = domains(URL)
    return dom.apply(lambda x: len(x & POPULAR_DOMAINS))

@feature
def popular_domain_in_subdomains(URL):
    sub = subdomain(URL)
    return sub.apply(lambda x: len(x & POPULAR_DOMAINS))
