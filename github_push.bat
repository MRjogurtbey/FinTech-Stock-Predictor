@echo off
chcp 65001 > nul
cd /d "C:\Users\Tuna\Desktop\Okul\4.sınıf.2.dönem\Finansta Bilişim Teknolojileri\Proje"

echo ============================================
echo  1/5  FinTech-Stock-Predictor siliniyor...
echo ============================================
rd /s /q "FinTech-Stock-Predictor" 2>nul
echo Tamam.

echo.
echo ============================================
echo  2/5  Eski .git temizleniyor...
echo ============================================
rd /s /q ".git" 2>nul
echo Tamam.

echo.
echo ============================================
echo  3/5  Git repo baslatiliyor...
echo ============================================
git init
git add .
git commit -m "Ilk surum: AI tabanli FinTech hisse tahmini projesi"
git branch -M main
git remote add origin https://github.com/MRjogurtbey/FinTech-Stock-Predictor

echo.
echo ============================================
echo  4/5  GitHub'a push ediliyor...
echo       (GitHub kullanici adi ve yeni PAT
echo        token girilmesi gerekebilir)
echo ============================================
git push -u origin main

echo.
echo ============================================
echo  5/5  TAMAMLANDI!
echo ============================================
pause
