MIT License

Copyright (c) 2025 Baha Büyükateş

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Drumveil Ritual: Metal Müzik için Derin Öğrenme ile Davul Nota Çıkarımı

Drumveil Ritual, metal müzik odaklı bir derin öğrenme projesi olarak, Slakh veri setindeki davul seslerini kullanarak bir "Onsets and Frames" modelini eğitmeyi ve bu türün yoğun ritimlerinden davul notalarını çıkarmayı hedefliyor. Proje, PyTorch ile geliştiriliyor ve Demucs’ü entegre ederek sesi davul, gitar ve vokal gibi bileşenlere ayırıyor, ardından spektrogram tabanlı bir yaklaşımla MIDI formatında nota çıktıları üretiyor.

Şu anda başlangıç aşamasında olan proje, metal müziğin agresif ve karmaşık yapısını analiz etme konusunda büyük bir potansiyel taşıyor. Eğitim sürecinde boyut uyumsuzlukları ve nota dengesizlikleri gibi çeşitli teknik zorluklarla karşılaşıyoruz, ancak bunlar aşılabilecek engeller. Demucs ile ses ayrıştırması, modelin davul vuruşlarını daha iyi izole etmesini sağlarken, "Onsets and Frames" mimarisi, metalin kaotik ritimlerini öğrenmek için sağlam bir temel sunuyor.

Nasıl İşleyecek? Slakh’den alınan drum track’leri ile model eğitilecek, Demucs sayesinde kullanıcıdan gelen metal şarkılarında davul izole edilecek ve model, bu izole edilmiş davul sinyalinden notaları çıkararak MIDI dosyası oluşturacak. Proje, metal müzik odaklıdır (örneğin, Sleep Token, Architects gibi gruplar hedeflenmektedir).

Proje henüz yolun başında ve önünde uzun bir geliştirme süreci var. Metal müziğin karmaşık doğası, daha fazla veri, ince ayar ve optimizasyon gerektiriyor.
