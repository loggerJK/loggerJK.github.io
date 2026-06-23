
$(document).ready(function() {
    $('.publication-mousecell').mouseover(function() {
        var $video = $(this).find('video');
        var $img = $(this).find('img');
        $video.css('display', 'inline-block');
        $img.css('display', 'none');
        if ($video.length) {
            var v = $video[0];
            v.muted = true;
            var p = v.play();
            if (p !== undefined) p.catch(function() {});
        }
    });
    $('.publication-mousecell').mouseout(function() {
        var $video = $(this).find('video');
        if ($video.length) $video[0].pause();
        $video.css('display', 'none');
        $(this).find('img').css('display', 'inline-block');
    });

    function prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    // Teaser lightbox: click (or Enter/Space) a publication thumbnail to enlarge it in-page.
    var $lightbox = $('#teaser-lightbox');
    var $lightboxContent = $lightbox.find('.teaser-modal-content');

    function openLightbox($cell) {
        var $video = $cell.find('video');
        var $img = $cell.find('img');
        var $media;
        if ($video.length) {
            var src = $video.find('source').attr('src') || $video.attr('src');
            $media = $('<video autoplay loop muted playsinline controls></video>')
                .attr('src', src)
                .on('error', function() {
                    $(this).replaceWith($('<div class="lightbox-error">Video unavailable.</div>'));
                });
        } else {
            $media = $('<img>')
                .attr('src', $img.attr('src'))
                .attr('alt', $img.attr('alt') || '');
        }
        $lightboxContent.empty().append($media);
        $lightbox.addClass('is-open').attr('aria-hidden', 'false');
        $('body').css('overflow', 'hidden');
    }

    function closeLightbox() {
        if (!$lightbox.hasClass('is-open')) return;
        $lightbox.removeClass('is-open').attr('aria-hidden', 'true');
        $lightboxContent.empty(); // stops/cleans up any injected video
        $('body').css('overflow', '');
    }

    $('.publication-image').click(function() {
        openLightbox($(this));
    });
    $('.publication-image').keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openLightbox($(this));
        }
    });
    $lightbox.find('.teaser-modal-bg, .teaser-modal-close').click(closeLightbox);
    $(document).keydown(function(e) {
        if (e.key === 'Escape') closeLightbox();
    });

    // Toggle news visibility
    var newsExpanded = false;
    $('#toggle-news-btn').click(function() {
        newsExpanded = !newsExpanded;
        var instant = prefersReducedMotion();
        if (newsExpanded) {
            if (instant) { $('.news-old').show(); } else { $('.news-old').slideDown(150); }
            $(this).addClass('expanded');
            $(this).find('span:last-child').text('Show Less');
        } else {
            if (instant) { $('.news-old').hide(); } else { $('.news-old').slideUp(150); }
            $(this).removeClass('expanded');
            $(this).find('span:last-child').text('Show All News');
        }
    });
})
