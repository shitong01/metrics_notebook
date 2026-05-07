grading_translate = {
                    "Hasarsız": "Undamaged",
                    "Bina Kilitli İnceleme Yapılamadı \(Girileme": "Unable to enter",
                    "Bina Kilitli İnceleme Yapılamadı (Girileme": "Unable to enter",
                    "Az Hasarlı": "Slightly damaged",
                    "Ağır Hasarlı": "Heavily damaged",
                    "Tespit Yapılamadı": "Could not detect",
                    "Acil Yıktırılacak": "To be demolished urgently",
                    "Değerlendirme Dışı": "Out of scope",
                    "Yıkık": "Collapsed",
                    "Tespit Edilemedi (Girilemedi)": "Unable to enter",
                    "Kapsam Dışı": "Out of scope",
                    }

# Map damage grades to numbers
#  assume 'unable to enter' is placed above 'slightly damaged', based on my plots of the distribution of DPM values
#  the buildings in those categories have
grading_score = {
                    "Hasarsız": 0,
                    "Bina Kilitli İnceleme Yapılamadı \(Girileme": 2,
                    "Bina Kilitli İnceleme Yapılamadı (Girileme": 2,
                    "Az Hasarlı": 1,
                    "Ağır Hasarlı": 3,
                    "Tespit Yapılamadı": 2,
                    "Acil Yıktırılacak": 4,
                    "Değerlendirme Dışı": 2,
                    "Yıkık": 5,
                    "Tespit Edilemedi (Girilemedi)": 2,
                    "Kapsam Dışı": 2,
                    }

accessible_categories = ['Undamaged', 'Slightly damaged', 'Heavily damaged', 'Heavily Damaged',
                    'To be demolished urgently', 'Collapsed']
damaged_categories = ['Ağır Hasarlı', 'Acil Yıktırılacak', 'Yıkık', 'Collapsed', 'To be demolished urgently',
                      'Heavily Damaged', 'Heavily damaged',
                      'Seriously damaged and destroyed, requiring immediate demolition']
inaccessible_categories = ['Unable to enter', 'Out of scope', 'Could not detect']

turkish_to_latin = {'Ü': 'U',
                    'Ö': 'O',
                    'İ': 'I',
                    'Â': 'A',
                    'Ğ': 'G',
                    'Ş': 'S',
                    'Ç': 'C',
                    }
