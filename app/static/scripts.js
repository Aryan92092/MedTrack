document.addEventListener('DOMContentLoaded',()=>{
  const flashes=document.querySelectorAll('.flash');
  setTimeout(()=>flashes.forEach(f=>f.remove()),4000);

  // Theme toggle persistence
  const root=document.documentElement;
  const toggle=document.getElementById('themeToggle');
  const getTheme=()=>root.getAttribute('data-theme')||'light';
  const setTheme=(t)=>{root.setAttribute('data-theme',t);try{localStorage.setItem('theme',t);}catch(e){}}
  if(toggle){
    toggle.addEventListener('click',()=>{
      const next=getTheme()==='dark'?'light':'dark';
      setTheme(next);
    });
  }

  // Button hover ripple effect
  document.querySelectorAll('button.btn').forEach(btn=>{
    btn.style.overflow='hidden';
    btn.addEventListener('pointerenter',()=>{btn.animate([{transform:'translateY(-1px)'},{transform:'translateY(-1px)'}],{duration:150,fill:'forwards'})});
    btn.addEventListener('pointerleave',()=>{btn.animate([{transform:'translateY(0)'},{transform:'translateY(0)'}],{duration:120,fill:'forwards'})});
  });
});


