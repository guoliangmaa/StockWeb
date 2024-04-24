"""
URL configuration for StockWeb project.

The `urlpatterns` list routes URLs to page_views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function page_views
    1. Add an import:  from my_app import page_views
    2. Add a URL to urlpatterns:  path('', page_views.home, name='home')
Class-based page_views
    1. Add an import:  from other_app.page_views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from .page_views import test_view

base_url = "api/"
urlpatterns = [
    path("admin/", admin.site.urls),
    re_path("api/test", test_view.TestView.as_view())
]
