from django.contrib import admin
from django.urls import path
from cache_app.views import dashboard_view, api_simulate, api_cache_get

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dashboard_view, name='dashboard'),
    path('api/simulate/', api_simulate, name='api_simulate'),
    path('api/cache/get/', api_cache_get, name='api_cache_get'),
]
