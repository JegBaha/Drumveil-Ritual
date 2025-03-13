

Drumveil Ritual: 
Derin Öğrenme ile Davul Nota Çıkarımı Özet Bu çalışma, SLKH (Synthesized Lakh) veri setindeki davul seslerini kullanarak bir "Onsets and Frames" derin öğrenme modelini eğitmeyi ve kullanıcı tarafından sağlanan bir şarkıdan davul notalarını çıkarmayı amaçlamaktadır. 
Proje, Drumveil Ritual adıyla anılmıştır; bu isim, davul ritimlerinin gizemli bir şekilde ortaya çıkarılmasını ve ritüelistik bir öğrenme sürecini simgelemektedir. 
Model, PyTorch ile geliştirilmiş, spektrogram tabanlı bir girişle çalışmakta ve MIDI formatında nota çıktıları üretmektedir. 
Eğitim sürecinde karşılaşılan boyut uyumsuzlukları ve nota dağılımındaki dengesizlikler giderilmiş, sonuç olarak dengeli bir davul nota çıkarımı elde edilmiştir.

Giriş 
Davul ritimleri, müzik prodüksiyonunda ve analizinde temel bir rol oynar. Ancak, bir şarkıdan davul notalarını otomatik olarak çıkarmak, karmaşık sinyal işleme ve makine öğrenimi teknikleri gerektirir. 
Bu proje, SLKH veri setindeki davul seslerini (drum.flac/wav) ve ilgili MIDI dosyalarını (drum.mid) kullanarak bir derin öğrenme modeli eğitmeyi hedeflemiştir. 
Eğitilen model, kullanıcı tarafından sağlanan bir şarkıdan (örneğin, drums.mp3) davul notalarını çıkararak MIDI formatında bir çıktı üretmektedir. 
Projenin temel katkıları şunlardır:

SLKH veri setiyle eğitilmiş bir "Onsets and Frames" modelinin geliştirilmesi. Spektrogram tabanlı bir girişle davul nota çıkarımının gerçekleştirilmesi. Eğitim sürecindeki boyut uyumsuzluklarının ve nota dağılımı dengesizliklerinin çözülmesi.

Sonuç Drumveil Ritual, SLKH veri setindeki davul seslerini kullanarak bir "Onsets and Frames" derin öğrenme modelini başarıyla eğitmiş ve kullanıcı şarkılarından davul notalarını çıkarmıştır. 
Eğitim sürecindeki boyut uyumsuzlukları ve nota dağılımındaki dengesizlikler giderilmiş, sonuç olarak dengeli bir nota çıkarımı elde edilmiştir. 
Gelecekte, veri artırımı, dinamik zaman boyutu ayarı ve daha uzun eğitim süreleri ile modelin performansı daha da iyileştirilebilir. 
Bu proje, müzik sinyal işleme ve derin öğrenme alanlarında otomatik nota çıkarımı için önemli bir adım teşkil etmektedir.
