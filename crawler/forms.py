from django import forms


class CrawlJobForm(forms.Form):
    urls = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 4,
                "placeholder": "https://example.com",
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white placeholder-gray-500 focus:outline-none focus:border-white transition-colors",
            }
        ),
        label="URLs (one per line)",
    )
    strategy = forms.ChoiceField(
        choices=[
            ("single_page", "Single URL"),
            ("sitemap", "Sitemap Crawl"),
            ("recursive", "Recursive Crawl"),
        ],
        widget=forms.Select(
            attrs={
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
            }
        ),
        initial="single_page",
        label="Crawl Strategy",
    )
    max_pages = forms.IntegerField(
        initial=10,
        min_value=1,
        label="Max Pages",
        widget=forms.NumberInput(
            attrs={
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
            }
        ),
    )
    max_depth = forms.IntegerField(
        initial=3,
        min_value=1,
        label="Max Depth",
        widget=forms.NumberInput(
            attrs={
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors"
            }
        ),
    )
    delay_between_requests = forms.FloatField(
        initial=0.5,
        min_value=0,
        label="Delay (seconds)",
        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "class": "mt-1 w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-sm text-white focus:outline-none focus:border-white transition-colors",
            }
        ),
    )
