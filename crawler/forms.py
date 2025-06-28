from django import forms
from .models import CrawlJob


class CrawlJobForm(forms.ModelForm):
    urls = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 4,
                "placeholder": "https://example.com",
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white placeholder-gray-500 focus:outline-none focus:border-white transition-colors",
            }
        ),
        label="URLs (one per line)",
        help_text="Enter one URL per line.",
    )

    class Meta:
        model = CrawlJob
        fields = [
            "strategy",
            "max_pages",
            "max_depth",
            "delay_between_requests",
        ]
        widgets = {
            "strategy": forms.Select(
                attrs={
                    "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
                }
            ),
            "max_pages": forms.NumberInput(
                attrs={
                    "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
                }
            ),
            "max_depth": forms.NumberInput(
                attrs={
                    "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
                }
            ),
            "delay_between_requests": forms.NumberInput(
                attrs={
                    "step": "0.1",
                    "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["strategy"].label = "Crawl Strategy"
        self.fields["max_pages"].label = "Max Pages"
        self.fields["max_depth"].label = "Max Depth"
        self.fields["delay_between_requests"].label = "Delay (seconds)"
        self.fields["strategy"].initial = "single"
        self.fields["max_pages"].initial = 10
        self.fields["max_depth"].initial = 3
        self.fields["delay_between_requests"].initial = 0.5
