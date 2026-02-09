import re
import tldextract
import pandas as pd
import pycountry


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
def domain_length(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: len(x.domain) + len(x.suffix) +  len(x.subdomain))

@feature
def domain_url_length_ratio(URL):
    ext = _ext_series(URL)
    return ext.apply(lambda x: (len(x.domain) + len(x.suffix) +  len(x.subdomain)) / len(URL)) 

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

@feature 
def has_org(URL):
    return suffix(URL).apply(lambda x: int('org' in x))

freaky_tlds = {'xyz', 'online', 'top', 'shop', 'site', 'icu', 'store', 'cyou', 'vip', 'live'}

@feature 
def num_freaky_tld(URL):
    return suffix(URL).apply(lambda x: len(x & freaky_tlds))


country_tlds = {
     country.alpha_2.lower()
    for country in pycountry.countries
}

@feature
def num_country_tld_in_subdomain(URL):
    return subdomain(URL).apply(lambda x: len(x & set(country_tlds)))
    
@feature
def num_country_tld_in_domain(URL):
    return domains(URL).apply(lambda x: len(x & set(country_tlds)))


@feature
def num_country_tld_in_suffix(URL):
    return suffix(URL).apply(lambda x: len(x & set(country_tlds)))


# trigger_words = {'login', 'bank', 'verify', 'secure', 'online', 'confirmation', 'payment'}
# @feature
# def trigger_words(URL)

popular_websites_df = pd.read_csv('OtherData/ranked_domains.csv')[['Domain']]

ext_popular = popular_websites_df['Domain'].apply(_extractor)

popular_domains = (ext_popular.apply(lambda x: _split(x.domain)).explode().dropna().unique())
popular_suffixs = (ext_popular.apply(lambda x: _split(x.suffix)).explode().dropna().unique())


POPULAR_DOMAINS = set(popular_domains)
POPULAR_DOMAINS.discard('www')
POPULAR_SUFFIXS = set(popular_suffixs)

@feature
def popular_domain_in_domain(URL):
    dom = domains(URL)
    return dom.apply(lambda x: len(x & POPULAR_DOMAINS))

@feature
def popular_domain_in_subdomain(URL):
    sub = subdomain(URL)
    return sub.apply(lambda x: len(x & POPULAR_DOMAINS))

@feature 
def popular_suffix_in_domain(URL):
    return domains(URL).apply(lambda x: len(x & POPULAR_SUFFIXS))


@feature 
def popular_suffix_in_subdomain(URL):
    return subdomain(URL).apply(lambda x: len(x & POPULAR_SUFFIXS))
    