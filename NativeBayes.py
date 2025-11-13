from collections import defaultdict, Counter
import math

def is_number(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def safe_log(x): return math.log(x) if x>0 else -math.inf

class NaiveBayesSimple:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.mode = None
        self.class_counts = Counter()
        self.class_priors = {}
        # tabular
        self.cond_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_vals = defaultdict(set)
        self.numeric_vals = defaultdict(lambda: defaultdict(list))
        self.numeric_stats = defaultdict(lambda: defaultdict(dict))
        # text
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)
        self.vocab = set()
        self.n = 0

    def fit(self, X, y):
        if len(X)!=len(y): raise ValueError("X and y same length")
        self.n = len(y)
        if self.n==0: raise ValueError("empty data")
        first = X[0]
        self.mode = 'text' if isinstance(first, str) else 'tabular'
        self.class_counts = Counter()
        self.cond_counts.clear(); self.feature_vals.clear()
        self.numeric_vals.clear(); self.numeric_stats.clear()
        self.word_counts.clear(); self.total_words.clear(); self.vocab.clear()

        for xi, lbl in zip(X, y):
            self.class_counts[lbl]+=1
            if self.mode=='text':
                for w in xi.lower().split():
                    self.word_counts[lbl][w]+=1; self.total_words[lbl]+=1; self.vocab.add(w)
            else:
                for f,v in xi.items():
                    if is_number(v):
                        self.numeric_vals[lbl][f].append(v)
                        self.feature_vals[f].add('<<num>>')
                    else:
                        self.cond_counts[lbl][f][v]+=1
                        self.feature_vals[f].add(v)

        self.class_priors = {c:cnt/self.n for c,cnt in self.class_counts.items()}
        if self.mode=='tabular':
            for lbl, feats in self.numeric_vals.items():
                for f, vals in feats.items():
                    m = sum(vals)/len(vals)
                    var = sum((z-m)**2 for z in vals)/len(vals) if vals else 0.0
                    self.numeric_stats[lbl][f] = {'mean':m, 'std':math.sqrt(var)}

    def _cond_cat(self, f, v, lbl, alpha):
        count = self.cond_counts[lbl][f].get(v,0)
        total = sum(self.cond_counts[lbl][f].values())
        k = len([x for x in self.feature_vals[f] if x!='<<num>>']) or 1
        if alpha==0:
            return count/total if total>0 else 0.0
        return (count+alpha)/(total+alpha*k)

    def _cond_word(self, w, lbl, alpha):
        cnt = self.word_counts[lbl].get(w,0); tot=self.total_words[lbl]; V=len(self.vocab) or 1
        if alpha==0: return cnt/tot if tot>0 else 0.0
        return (cnt+alpha)/(tot+alpha*V)

    def _gauss(self, x, mean, std):
        if std==0: return 1.0 if x==mean else 1e-9
        return (1.0/(math.sqrt(2*math.pi)*std))*math.exp(-((x-mean)**2)/(2*std*std))

    def predict_proba(self, x, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        if self.mode is None: raise ValueError("fit first")
        logs = {}
        for lbl in self.class_counts:
            prior = self.class_priors.get(lbl,0)
            logs[lbl] = safe_log(prior) if prior>0 else -math.inf
            if logs[lbl]==-math.inf: continue
            if self.mode=='text':
                for w in x.lower().split():
                    p=self._cond_word(w,lbl,alpha)
                    if p==0: logs[lbl]=-math.inf; break
                    logs[lbl]+=safe_log(p)
            else:
                for f,v in x.items():
                    if is_number(v):
                        stats = self.numeric_stats.get(lbl,{}).get(f,None)
                        p = self._gauss(v, stats['mean'], stats['std']) if stats else 1e-9
                    else:
                        p=self._cond_cat(f,v,lbl,alpha)
                    if p==0: logs[lbl]=-math.inf; break
                    logs[lbl]+=safe_log(p)
        finite = [v for v in logs.values() if v!=-math.inf]
        if not finite: return {lbl:0.0 for lbl in self.class_counts}
        M=max(finite)
        exps={lbl:(0.0 if logs[lbl]==-math.inf else math.exp(logs[lbl]-M)) for lbl in logs}
        S=sum(exps.values()) or 1.0
        return {lbl: exps[lbl]/S for lbl in exps}

    def predict(self, X, alpha=None):
        single = False
        if not isinstance(X, list): X=[X]; single=True
        preds=[max(self.predict_proba(x,alpha), key=lambda k: self.predict_proba(x,alpha)[k]) for x in X]
        return preds[0] if single else preds

# === Example: đổi X,y tuỳ ý ===
if __name__=='__main__':
    # Tabular example:
    X = [{"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Weak"},
         {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Strong"},
         {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"},
         {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"},
         {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak"}

         ]

    y = ["No","No","No","Yes","No"]
    nb = NaiveBayesSimple(alpha=1.0); nb.fit(X,y)
    print(nb.predict_proba({"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Wind":"Weak"}))

    # # Text example:
    # X2 = ["mua ngay giảm giá","bạn có thời gian không"]
    # y2 = ["spam","ham"]
    # nb2 = NaiveBayesSimple(alpha=1.0); nb2.fit(X2,y2)
    # print(nb2.predict_proba("giảm giá hôm nay"))
