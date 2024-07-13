"""
URL configuration for StockWeb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from .views import test_view, recommend_stock_view, tts_view, warning_stock_view
# 配置静态和媒体文件的访问
from django.conf import settings
from django.conf.urls.static import static

base_url = "api/"
urlpatterns = [
    path("admin/", admin.site.urls),
    re_path("api/test", test_view.TestView.as_view()),
    path("api/stock/recommend", recommend_stock_view.RecommendStockView.as_view()),
    path("api/stock/tts", tts_view.TTSView.as_view()),
    path("api/stock/warning", warning_stock_view.WarningStockView.as_view()),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)