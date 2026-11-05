"""Portable extraction of the frozen F1 atom_sets kernel from f1_causal_repairs.py.
The atom-creation rules below are retained. No reference assertion is read.
"""
import re
TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?:\d+\.\d+|\d+)|[A-Za-z_$][\w$]*|==|!=|<=|>=|&&|\|\||[^\s]')
GENERIC={'0','1','-1','true','false','null','==','!=','>','<','>=','<=','+','-','*','/','%','!','&&','||','(',')','[',']',',','.', '?', ':'}
METHODS=set('size length isEmpty get set contains containsKey containsValue getValue getKey equals hashCode toString iterator next hasNext stream count isPresent getProperty getName getMessage getCause getClass booleanValue intValue longValue doubleValue floatValue valueOf compareTo asList emptyList emptyMap emptySet is equalTo not hasItem hasItems hasSize containsString startsWith endsWith nullValue notNullValue instanceOf allOf anyOf sameInstance greaterThan lessThan closeTo'.split())
framework=set('org junit Assert hamcrest Matchers CoreMatchers Is java util lang Arrays Collections Integer Long Double Float Boolean Object String StringBuilder Math BigDecimal List Set Map Collection Optional Stream Collectors'.split())
operators={'instanceof'}
constants={'""',"''",'0.0','1.0','0L','1L','0l','1l'}
ident=re.compile(r'^[A-Za-z_$][\w$]*$')
num=re.compile(r'^\d+$')
def toks(s): return TOKEN.findall(s)
def atom_sets(source):
 ts=toks(source); st=set(ts)
 base=st|GENERIC|METHODS
 p1=base|framework|operators|constants
 numbers={int(t) for t in st if num.fullmatch(t) and len(t)<=7 and int(t)<=1000000}
 p2=p1|{str(v+d) for v in numbers for d in (-1,1) if 0<=v+d<=1000000}
 if any(t in st for t in ('for','while','length','size','get','List','Array','array')):
  p2|={'2','3'}
 p3=set(p2)
 for t in st:
  if ident.fullmatch(t) and 1<len(t)<45 and not t[0].isupper() and not t.startswith(('get','set','is','has')):
   cap=t[0].upper()+t[1:]
   if any(z in st for z in ('get','getValue','getName','getProperty','getId','set','isEmpty')):
    p3.add('get'+cap)
 return [base,p1,p2,p3]

